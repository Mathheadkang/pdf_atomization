"""Export endpoints for generating markdown files."""

import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import settings
from app.models import (
    ExportRequest,
    ExportResponse,
    JobStatus
)
from app.routers.upload import get_job
from app.services.markdown_generator import MarkdownGenerator

router = APIRouter()


@router.post("/export/{job_id}", response_model=ExportResponse)
async def export_to_folder(
    job_id: str,
    include_filtered: bool = False
) -> ExportResponse:
    """Export processed PDF to markdown files in output folder."""

    job = get_job(job_id)

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot export. Job status: {job.status.value}"
        )

    if not job.structure:
        raise HTTPException(
            status_code=500,
            detail="Structure not available"
        )

    # Generate markdown files
    generator = MarkdownGenerator(settings.output_dir)
    generated_files = await generator.generate_files(
        job.structure,
        include_filtered=include_filtered
    )

    output_path = generator.get_output_path(job.structure)

    # Resolve any remaining placeholder links
    link_stats = await generator.link_manager.resolve_all_links(output_path)

    return ExportResponse(
        job_id=job_id,
        output_path=str(output_path),
        message=f"Exported {len(generated_files)} files to {output_path} (resolved {link_stats['links_resolved']} links)"
    )


@router.get("/export/{job_id}/download")
async def download_as_zip(
    job_id: str,
    include_filtered: bool = False
) -> FileResponse:
    """Export and download as ZIP file."""

    job = get_job(job_id)

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot export. Job status: {job.status.value}"
        )

    if not job.structure:
        raise HTTPException(
            status_code=500,
            detail="Structure not available"
        )

    # Generate to temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_output = Path(temp_dir) / "output"
        temp_output.mkdir()

        generator = MarkdownGenerator(temp_output)
        await generator.generate_files(
            job.structure,
            include_filtered=include_filtered
        )

        output_path = generator.get_output_path(job.structure)

        # Resolve any remaining placeholder links
        await generator.link_manager.resolve_all_links(output_path)

        # Create ZIP
        zip_name = f"{job.structure.title.replace(' ', '_')}"
        zip_path = Path(temp_dir) / zip_name

        shutil.make_archive(
            str(zip_path),
            'zip',
            output_path
        )

        zip_file = Path(f"{zip_path}.zip")

        # Copy to uploads dir for serving
        final_zip = settings.uploads_dir / f"{job_id}.zip"
        shutil.copy(zip_file, final_zip)

    return FileResponse(
        path=final_zip,
        filename=f"{zip_name}.zip",
        media_type="application/zip"
    )


@router.get("/export/{job_id}/files")
async def list_exported_files(job_id: str) -> dict:
    """List all files that would be generated for an export."""

    job = get_job(job_id)

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot list files. Job status: {job.status.value}"
        )

    if not job.structure:
        raise HTTPException(
            status_code=500,
            detail="Structure not available"
        )

    # Build file list from structure
    files = []

    def collect_files(node, path_prefix=""):
        if not node.included:
            return

        # Determine filename
        name = node.title.replace(" ", "_")[:50]
        name = ''.join(c for c in name if c.isalnum() or c in '_-')

        if node.type.value in ["book", "chapter"]:
            file_path = f"{path_prefix}{name}/index.md"
            new_prefix = f"{path_prefix}{name}/"
        else:
            file_path = f"{path_prefix}{name}.md"
            new_prefix = path_prefix

        files.append({
            "path": file_path,
            "title": node.title,
            "type": node.type.value,
            "atom_type": node.atom_type.value if node.atom_type else None,
            "is_atomic": node.atomization_status.value if hasattr(node, 'atomization_status') else None
        })

        for child in node.children:
            collect_files(child, new_prefix)

    # Add root index
    files.append({
        "path": "index.md",
        "title": job.structure.title,
        "type": "book"
    })

    for child in job.structure.root.children:
        collect_files(child)

    return {
        "job_id": job_id,
        "file_count": len(files),
        "files": files
    }
