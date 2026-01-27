#!/usr/bin/env python3
"""
Script to check memory_mode status of agents in the database.

Usage:
    python scripts/check_memory_mode_status.py [--agent-id AGENT_ID] [--all] [--migrate-dry-run]

Examples:
    # Check status of all agents
    python scripts/check_memory_mode_status.py --all

    # Check status of a specific agent
    python scripts/check_memory_mode_status.py --agent-id agent-123

    # Show what would be migrated (dry run)
    python scripts/check_memory_mode_status.py --all --migrate-dry-run
"""

import argparse
import asyncio
import sys
from collections import defaultdict
from typing import Optional

# Add the parent directory to the path for imports
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select, func

from letta.orm.agent import Agent as AgentModel
from letta.orm.sqlalchemy_base import get_db_registry
from letta.schemas.llm_config import LLMConfig


async def check_agent_memory_mode(agent_id: str) -> dict:
    """Check memory_mode status for a specific agent."""
    db_registry = get_db_registry()
    
    async with db_registry.async_session() as session:
        result = await session.execute(
            select(AgentModel).where(AgentModel.id == agent_id)
        )
        agent = result.scalar_one_or_none()
        
        if agent is None:
            return {"error": f"Agent {agent_id} not found"}
        
        supports_developer = LLMConfig.supports_developer_role(agent.llm_config) if agent.llm_config else False
        
        return {
            "agent_id": agent.id,
            "name": agent.name,
            "memory_mode": agent.memory_mode,
            "model": agent.llm_config.model if agent.llm_config else None,
            "model_endpoint_type": agent.llm_config.model_endpoint_type if agent.llm_config else None,
            "supports_developer_role": supports_developer,
            "needs_migration": agent.memory_mode is None,
            "would_migrate_to": "context_message" if supports_developer else "system_prompt",
            "num_messages": len(agent.message_ids) if agent.message_ids else 0,
        }


async def check_all_agents_memory_mode(migrate_dry_run: bool = False) -> dict:
    """Check memory_mode status for all agents."""
    db_registry = get_db_registry()
    
    async with db_registry.async_session() as session:
        # Get summary counts
        result = await session.execute(
            select(
                AgentModel.memory_mode,
                func.count(AgentModel.id)
            ).group_by(AgentModel.memory_mode)
        )
        mode_counts = {row[0] or "None (not evaluated)": row[1] for row in result.fetchall()}
        
        # Get all agents for detailed analysis
        result = await session.execute(select(AgentModel))
        agents = result.scalars().all()
        
        # Categorize agents
        stats = {
            "total_agents": len(agents),
            "memory_mode_counts": mode_counts,
            "by_model_endpoint_type": defaultdict(lambda: {"total": 0, "needs_migration": 0}),
            "migration_preview": [],
        }
        
        for agent in agents:
            endpoint_type = agent.llm_config.model_endpoint_type if agent.llm_config else "unknown"
            stats["by_model_endpoint_type"][endpoint_type]["total"] += 1
            
            if agent.memory_mode is None:
                stats["by_model_endpoint_type"][endpoint_type]["needs_migration"] += 1
                
                supports_developer = LLMConfig.supports_developer_role(agent.llm_config) if agent.llm_config else False
                would_migrate_to = "context_message" if supports_developer else "system_prompt"
                
                if migrate_dry_run:
                    stats["migration_preview"].append({
                        "agent_id": agent.id,
                        "name": agent.name,
                        "model": agent.llm_config.model if agent.llm_config else None,
                        "endpoint_type": endpoint_type,
                        "would_migrate_to": would_migrate_to,
                    })
        
        return stats


def print_agent_status(status: dict):
    """Pretty print a single agent's status."""
    if "error" in status:
        print(f"\n❌ {status['error']}")
        return
    
    print(f"\n{'='*60}")
    print(f"Agent: {status['name']} ({status['agent_id']})")
    print(f"{'='*60}")
    print(f"  Memory Mode:           {status['memory_mode'] or 'None (not evaluated)'}")
    print(f"  Model:                 {status['model']}")
    print(f"  Endpoint Type:         {status['model_endpoint_type']}")
    print(f"  Supports Developer:    {'✅ Yes' if status['supports_developer_role'] else '❌ No'}")
    print(f"  Needs Migration:       {'⚠️  Yes' if status['needs_migration'] else '✅ No'}")
    if status['needs_migration']:
        print(f"  Would Migrate To:      {status['would_migrate_to']}")
    print(f"  Message Count:         {status['num_messages']}")


def print_all_agents_status(stats: dict, migrate_dry_run: bool):
    """Pretty print status for all agents."""
    print(f"\n{'='*70}")
    print(f"MEMORY MODE STATUS REPORT")
    print(f"{'='*70}")
    
    print(f"\n📊 Summary:")
    print(f"  Total Agents: {stats['total_agents']}")
    
    print(f"\n📦 Memory Mode Distribution:")
    for mode, count in sorted(stats['memory_mode_counts'].items()):
        pct = (count / stats['total_agents'] * 100) if stats['total_agents'] > 0 else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {mode:25s} {bar} {count:5d} ({pct:5.1f}%)")
    
    print(f"\n🔌 By Model Endpoint Type:")
    for endpoint_type, data in sorted(stats['by_model_endpoint_type'].items()):
        needs = data['needs_migration']
        total = data['total']
        status = "⚠️" if needs > 0 else "✅"
        print(f"  {status} {endpoint_type:20s}: {total:4d} total, {needs:4d} need migration")
    
    if migrate_dry_run and stats['migration_preview']:
        print(f"\n🔄 Migration Preview (--migrate-dry-run):")
        print(f"  The following {len(stats['migration_preview'])} agent(s) would be migrated:\n")
        
        for item in stats['migration_preview'][:20]:  # Limit to first 20
            print(f"  • {item['name'][:30]:30s} ({item['agent_id'][:20]}...)")
            print(f"    Model: {item['model']}, Endpoint: {item['endpoint_type']}")
            print(f"    → Would migrate to: {item['would_migrate_to']}")
        
        if len(stats['migration_preview']) > 20:
            print(f"\n  ... and {len(stats['migration_preview']) - 20} more agents")
    
    # Summary recommendation
    total_needing_migration = sum(d['needs_migration'] for d in stats['by_model_endpoint_type'].values())
    if total_needing_migration > 0:
        print(f"\n⚠️  {total_needing_migration} agent(s) have memory_mode=None and will be")
        print(f"   auto-migrated on their next execution (agent.step()).")
        print(f"\n   Migration happens automatically when:")
        print(f"   - auto_migrate_memory_mode=True is passed to get_agent_by_id_async")
        print(f"   - This is done by LettaAgent/VoiceAgent during step() calls")
    else:
        print(f"\n✅ All agents have been evaluated for memory mode!")


async def main():
    parser = argparse.ArgumentParser(description="Check memory_mode status of agents")
    parser.add_argument("--agent-id", help="Check a specific agent by ID")
    parser.add_argument("--all", action="store_true", help="Check all agents")
    parser.add_argument("--migrate-dry-run", action="store_true", 
                        help="Show what would be migrated (requires --all)")
    
    args = parser.parse_args()
    
    if not args.agent_id and not args.all:
        parser.print_help()
        print("\n❌ Error: Please specify --agent-id or --all")
        sys.exit(1)
    
    if args.agent_id:
        status = await check_agent_memory_mode(args.agent_id)
        print_agent_status(status)
    
    if args.all:
        stats = await check_all_agents_memory_mode(args.migrate_dry_run)
        print_all_agents_status(stats, args.migrate_dry_run)


if __name__ == "__main__":
    asyncio.run(main())
