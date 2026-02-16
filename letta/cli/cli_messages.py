"""CLI commands for message operations."""

import asyncio
from datetime import datetime
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from letta.log import get_logger

logger = get_logger(__name__)
console = Console()

app = typer.Typer(name="messages", help="Message operations")


@app.command("backfill")
def backfill_messages(
    agent_id: Annotated[Optional[str], typer.Option(help="Filter messages by agent ID")] = None,
    batch_size: Annotated[int, typer.Option(help="Number of messages to process per batch")] = 100,
    dry_run: Annotated[bool, typer.Option(help="Show what would be done without actually doing it")] = False,
    force: Annotated[bool, typer.Option(help="Re-embed messages that already exist in Qdrant")] = False,
    start_date: Annotated[Optional[str], typer.Option(help="Only backfill messages after this date (YYYY-MM-DD)")] = None,
    end_date: Annotated[Optional[str], typer.Option(help="Only backfill messages before this date (YYYY-MM-DD)")] = None,
):
    """
    Backfill messages from PostgreSQL to Qdrant vector database.
    
    This command is useful when:
    - Setting up Qdrant for the first time
    - Migrating from Turbopuffer to Qdrant
    - Rebuilding the vector database after data loss
    
    Examples:
        # Backfill all messages (dry run first)
        letta messages backfill --dry-run
        letta messages backfill
        
        # Backfill for specific agent
        letta messages backfill --agent-id agent_abc123
        
        # Backfill with custom batch size
        letta messages backfill --batch-size 50
        
        # Force re-embedding of existing messages
        letta messages backfill --force
    """
    from letta.helpers.qdrant_client import QdrantClient, should_use_qdrant_for_messages
    from letta.services.user_manager import UserManager
    from letta.settings import settings

    # Check if Qdrant is configured
    if not should_use_qdrant_for_messages():
        console.print("[red]❌ Qdrant is not configured for messages[/red]")
        console.print("\nTo use Qdrant, set the following environment variables:")
        console.print("  - VECTOR_DB_PROVIDER=qdrant")
        console.print("  - QDRANT_URL=<your_qdrant_url>")
        console.print("  - EMBED_ALL_MESSAGES=true")
        console.print("  - OPENAI_API_KEY=<your_openai_key>")
        raise typer.Exit(code=1)

    # Parse dates if provided
    start_date_obj = None
    end_date_obj = None
    if start_date:
        try:
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError:
            console.print(f"[red]❌ Invalid start date format: {start_date}. Use YYYY-MM-DD[/red]")
            raise typer.Exit(code=1)
    
    if end_date:
        try:
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            console.print(f"[red]❌ Invalid end date format: {end_date}. Use YYYY-MM-DD[/red]")
            raise typer.Exit(code=1)

    async def run_backfill():
        # Get user manager and actor
        user_manager = UserManager()
        actor = await user_manager.get_default_user_async()
        
        if not actor:
            console.print("[red]❌ No default user found[/red]")
            raise typer.Exit(code=1)
        
        # Create Qdrant client
        try:
            client = QdrantClient()
        except Exception as e:
            console.print(f"[red]❌ Failed to create Qdrant client: {e}[/red]")
            raise typer.Exit(code=1)
        
        # Display configuration
        console.print("\n[bold]Backfill Configuration:[/bold]")
        console.print(f"  Organization: {actor.organization_id}")
        if agent_id:
            console.print(f"  Agent ID: {agent_id}")
        else:
            console.print("  Agent ID: [dim]All agents[/dim]")
        console.print(f"  Batch size: {batch_size}")
        console.print(f"  Force re-embed: {force}")
        if start_date_obj:
            console.print(f"  Start date: {start_date}")
        if end_date_obj:
            console.print(f"  End date: {end_date}")
        console.print(f"  Dry run: {dry_run}")
        console.print()
        
        # Run backfill
        try:
            if dry_run:
                console.print("[yellow]🔍 Running in dry-run mode...[/yellow]\n")
            else:
                console.print("[cyan]🚀 Starting backfill...[/cyan]\n")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Backfilling messages...", total=None)
                
                stats = await client.backfill_messages_for_org(
                    organization_id=actor.organization_id,
                    actor=actor,
                    agent_id=agent_id,
                    batch_size=batch_size,
                    force=force,
                    start_date=start_date_obj,
                    end_date=end_date_obj,
                    dry_run=dry_run,
                )
                
                progress.update(task, completed=True)
            
            # Display results
            console.print()
            if dry_run:
                console.print("[bold]Dry Run Results:[/bold]")
                console.print(f"  Total messages found: {stats['total_messages']}")
                if 'estimated_cost' in stats:
                    console.print(f"  Estimated cost: ${stats['estimated_cost']:.4f} (OpenAI embeddings)")
                console.print("\n[yellow]ℹ️  Run without --dry-run to actually backfill messages[/yellow]")
            else:
                # Create results table
                table = Table(title="Backfill Results", show_header=True)
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")
                
                table.add_row("Total messages", str(stats['total_messages']))
                table.add_row("Processed", str(stats['processed']))
                table.add_row("Skipped", str(stats['skipped']))
                table.add_row("Failed", str(stats['failed']))
                table.add_row("Batches", str(stats['batches']))
                table.add_row("Time taken", f"{stats['time_taken']:.2f}s")
                
                console.print(table)
                console.print()
                
                if stats['failed'] > 0:
                    console.print(f"[yellow]⚠️  {stats['failed']} messages failed to process[/yellow]")
                else:
                    console.print("[green]✅ Backfill completed successfully![/green]")
        
        except Exception as e:
            console.print(f"\n[red]❌ Backfill failed: {e}[/red]")
            logger.exception("Backfill failed")
            raise typer.Exit(code=1)
    
    # Run async function
    asyncio.run(run_backfill())


if __name__ == "__main__":
    app()
