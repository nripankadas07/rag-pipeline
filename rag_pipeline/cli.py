"""CLI interface for the RAG pipeline."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from .chunkers import CHUNKERS
from .pipeline import Pipeline

console = Console()


@click.group()
def cli() -> None:
    """RAG Pipeline — chunk, embed, retrieve, evaluate."""
    pass


@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--chunker", type=click.Choice(list(CHUNKERS.keys())), default="recursive")
@click.option("--chunk-size", default=512, help="Target chunk size in characters.")
@click.option("--json-output", is_flag=True)
def chunk(file_path: str, chunker: str, chunk_size: int, json_output: bool) -> None:
    """Chunk a document and show the results."""
    from .chunkers import get_chunker

    text = Path(file_path).read_text(encoding="utf-8")
    c = get_chunker(chunker, chunk_size=chunk_size)
    chunks = c.chunk(text)

    if json_output:
        import json
        output = [{"index": ch.index, "length": len(ch), "preview": ch.text[:100]} for ch in chunks]
        click.echo(json.dumps(output, indent=2))
    else:
        table = Table(title=f"Chunks ({chunker})")
        table.add_column("#", style="cyan", width=4)
        table.add_column("Length", style="green", width=8)
        table.add_column("Preview", style="white")
        for ch in chunks:
            table.add_row(str(ch.index), str(len(ch)), ch.text[:80] + "...")
        console.print(table)
        console.print(f"\n[bold]{len(chunks)}[/bold] chunks from {file_path}")


@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.argument("query")
@click.option("--k", default=5, help="Number of results to return.")
@click.option("--chunker", default="recursive")
@click.option("--json-output", is_flag=True)
def search(file_path: str, query: str, k: int, chunker: str, json_output: bool) -> None:
    """Ingest a document and search it."""
    text = Path(file_path).read_text(encoding="utf-8")
    pipe = Pipeline(chunker_name=chunker)
    num = pipe.ingest(text, source=file_path)

    result = pipe.query(query, k=k)

    if json_output:
        import json
        output = {
            "query": query,
            "chunks_ingested": num,
            "results": [
                {"score": round(r.score, 4), "text": r.text[:200]}
                for r in result.results
            ],
        }
        click.echo(json.dumps(output, indent=2))
    else:
        console.print(f"[cyan]Query:[/cyan] {query}")
        console.print(f"[dim]Ingested {num} chunks from {file_path}[/dim]\n")

        for i, r in enumerate(result.results, 1):
            console.print(f"[bold green]#{i}[/bold green] (score: {r.score:.4f})")
            console.print(f"  {r.text[:200]}...")
            console.print()
