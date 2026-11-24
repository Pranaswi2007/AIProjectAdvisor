"""
AI Project Intelligence & Risk Advisor — Flask Application
===========================================================
Milestone 1: Document ingestion, RAG pipeline, and knowledge-base search.
Milestone 2: Three-agent analysis pipeline (scope, blockers, risks).
Milestone 3: Conversational Q&A assistant.
Milestone 4: Insights dashboard, incremental upload, report persistence.

Routes
------
GET  /                        → Dashboard (project stats + last-analysis summaries)
GET  /upload                  → Upload page
POST /upload                  → Process uploaded files (incremental supported)
GET  /knowledge-base          → Browse indexed documents per project
POST /knowledge-base/delete   → Remove a document from the vector store
POST /knowledge-base/delete-project → Remove an entire project (all docs + disk folder)
GET  /search                  → Search form
POST /search                  → Execute RAG search and show results
GET  /analyse                 → Analysis project selector
POST /analyse                 → Run three-agent pipeline and show report
GET  /qa                      → Conversational Q&A chat page
POST /ask                     → Answer a question via RAG + LLM (JSON)
"""

import os
import uuid
from pathlib import Path

from flask import Flask, flash, jsonify, redirect, render_template, request, url_for

from ingestion import LOADERS, ingest_file
from rag import delete_document, get_model, get_project_stats, index_document, list_sources, search
from report_store import delete_report, list_summaries, save_report

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "dev-secret-change-in-prod")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB total per request

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {ext.lstrip(".") for ext in LOADERS}


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(413)
def too_large(e):
    flash("Upload too large. Total size must be under 100 MB.", "error")
    return redirect(url_for("upload")), 413


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _cleanup_old_projects(keep_project_id: str):
    """Delete all projects except keep_project_id — uploads, ChromaDB, reports.

    Called when uploading files for a new project so old project data doesn't
    accumulate on disk or in the vector store.
    """
    import shutil
    import logging

    # 1. Delete old upload folders
    if UPLOAD_DIR.exists():
        for folder in UPLOAD_DIR.iterdir():
            if not folder.is_dir():
                continue
            if folder.name == keep_project_id:
                continue  # keep the current project
            try:
                shutil.rmtree(str(folder), ignore_errors=True)
                logging.info("Cleaned up old project folder: %s", folder.name)
            except Exception as exc:
                logging.warning("Could not remove old folder '%s': %s", folder.name, exc)

    # 2. Delete old ChromaDB collections
    try:
        from rag import _get_client
        import re as _re
        client = _get_client()
        for collection in client.list_collections():
            # Collection names are "project-<sanitised_id>"
            name = collection.name
            if not name.startswith("project-"):
                continue
            # Extract the project ID from the collection name
            proj_id = name[8:]  # strip "project-"
            if proj_id == keep_project_id:
                continue
            try:
                client.delete_collection(name)
                logging.info("Cleaned up old ChromaDB collection: %s", name)
            except Exception as exc:
                logging.warning("Could not remove collection '%s': %s", name, exc)
    except Exception as exc:
        logging.warning("ChromaDB cleanup error: %s", exc)

    # 3. Delete old saved reports
    try:
        from report_store import delete_report
        reports_dir = Path("reports")
        if reports_dir.exists():
            for f in reports_dir.glob("*.json"):
                proj_id = f.stem
                if proj_id == keep_project_id:
                    continue
                delete_report(proj_id)
                logging.info("Cleaned up old report: %s", proj_id)
    except Exception as exc:
        logging.warning("Report cleanup error: %s", exc)


def get_all_projects() -> list[dict]:
    """Scan the uploads folder and build a list of project summaries.

    Skips folders that are completely empty (no files on disk and no indexed
    chunks in ChromaDB) so deleted projects don't ghost in the UI.

    Each project dict contains:
      - id, name, document_count, chunk_count, sources  (from ChromaDB)
      - last_analysis  — compact summary from report_store (or None)
      - stale          — True when the KB was updated after the last analysis
    """
    projects = []
    summaries = list_summaries()          # {project_id: summary_dict}

    if UPLOAD_DIR.exists():
        for folder in sorted(UPLOAD_DIR.iterdir()):
            if not folder.is_dir():
                continue
            stats = get_project_stats(folder.name)
            has_files = any(folder.iterdir())
            if not has_files and stats["chunk_count"] == 0:
                try:
                    folder.rmdir()
                except Exception:
                    pass
                continue

            summary = summaries.get(folder.name)

            # Stale flag: any file on disk newer than the last saved_at timestamp
            stale = False
            if summary:
                try:
                    from datetime import datetime as _dt
                    saved_at = _dt.fromisoformat(summary.get("saved_at", "2000-01-01"))
                    for f in folder.iterdir():
                        if f.is_file() and _dt.fromtimestamp(f.stat().st_mtime) > saved_at:
                            stale = True
                            break
                except Exception:
                    pass

            projects.append({
                "id":            folder.name,
                "name":          folder.name.replace("-", " ").title(),
                "last_analysis": summary,
                "stale":         stale,
                **stats,
            })
    return projects


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def dashboard():
    # Redirect to the new single-page application
    return redirect(url_for("spa"))


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "GET":
        return render_template("upload.html", allowed=sorted(ALLOWED_EXTENSIONS))

    # ---- POST: process uploaded files ----
    project_id = request.form.get("project_id", "").strip()
    if not project_id:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"error": "Please enter a project name."}), 400
        flash("Please enter a project name.", "error")
        return redirect(url_for("upload"))

    # Sanitise project_id — keep only alphanumeric, hyphens, underscores
    project_id = "".join(c if c.isalnum() or c in "-_" else "-" for c in project_id).strip("-")
    if not project_id:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"error": "Invalid project name. Use letters, numbers, hyphens or underscores."}), 400
        flash("Invalid project name. Use letters, numbers, hyphens or underscores.", "error")
        return redirect(url_for("upload"))

    # Optional document-type tag for incremental uploads
    doc_type = request.form.get("doc_type", "").strip()

    # ── Clean up old projects: keep only the current project ──────────
    # Run in background so it doesn't add latency to the upload response.
    import threading
    t = threading.Thread(target=_cleanup_old_projects, args=(project_id,), daemon=True)
    t.start()

    files = request.files.getlist("documents")
    if not files or all(f.filename == "" for f in files):
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"error": "No files selected."}), 400
        flash("No files selected.", "error")
        return redirect(url_for("upload"))

    project_dir = UPLOAD_DIR / project_id
    project_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for f in files:
        if f.filename == "":
            continue
        if not allowed_file(f.filename):
            results.append({"name": f.filename, "status": "error",
                             "message": f"Unsupported type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"})
            continue

        save_path = project_dir / f.filename

        # Duplicate-file detection: warn but still re-index (overwrites on disk)
        is_duplicate = save_path.exists()

        f.save(str(save_path))

        try:
            doc   = ingest_file(str(save_path), project_id)
            n     = index_document(doc)
            # NOTE: Do NOT clear the LLM result cache on upload. The cache key
            # includes an MD5 hash of the full document text — if the files are
            # the same, the hash matches and the cache hits (instant analysis).
            # If the files are different, the hash changes and the cache misses
            # automatically. Explicit clearing would destroy cached results
            # when re-uploading the same files for a presentation.
            msg = f"{n} chunks indexed  |  {doc.word_count:,} words"
            if is_duplicate:
                msg += "  (re-indexed — file already existed)"
            if doc_type:
                msg += f"  |  type: {doc_type}"
            results.append({
                "name":      f.filename,
                "status":    "success",
                "message":   msg,
                "duplicate": is_duplicate,
            })
        except Exception as exc:
            results.append({"name": f.filename, "status": "error",
                             "message": str(exc), "duplicate": False})

    # Flag whether this was an incremental update to an existing project
    is_incremental = any(r.get("duplicate") for r in results)

    # AJAX request from the single-page app → return JSON
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        stats = get_project_stats(project_id)
        return jsonify({
            "project_id":     project_id,
            "results":        results,
            "doc_type":       doc_type,
            "is_incremental": is_incremental,
            "stats":          stats,
        })

    return render_template("upload_result.html", project_id=project_id,
                           results=results, doc_type=doc_type,
                           is_incremental=is_incremental)


@app.route("/knowledge-base")
def knowledge_base():
    project_id = request.args.get("project_id", "").strip()
    projects   = get_all_projects()

    sources = {}
    if project_id:
        sources = list_sources(project_id)

    return render_template(
        "knowledge_base.html",
        projects=projects,
        selected_project=project_id,
        sources=sources,
    )


@app.route("/knowledge-base/delete", methods=["POST"])
def kb_delete():
    import stat
    project_id = request.form.get("project_id", "").strip()
    filename   = request.form.get("filename",   "").strip()

    if project_id and filename:
        deleted = delete_document(project_id, filename)
        disk_path = UPLOAD_DIR / project_id / filename
        if disk_path.exists():
            try:
                disk_path.unlink()
            except PermissionError:
                try:
                    disk_path.chmod(stat.S_IWRITE)
                    disk_path.unlink()
                except Exception:
                    flash(
                        f"Removed '{filename}' from the knowledge base ({deleted} chunks), "
                        "but the file on disk could not be deleted (it may be open elsewhere).",
                        "info",
                    )
                    return redirect(url_for("knowledge_base", project_id=project_id))
        flash(f"Removed '{filename}' ({deleted} chunks deleted).", "success")
    else:
        flash("Invalid delete request.", "error")

    return redirect(url_for("knowledge_base", project_id=project_id))


@app.route("/knowledge-base/delete-project", methods=["POST"])
def kb_delete_project():
    """Delete every document in a project — ChromaDB collection + all files on disk."""
    import shutil
    import stat
    import logging
    project_id = request.form.get("project_id", "").strip()

    if not project_id:
        flash("Invalid request: no project specified.", "error")
        return redirect(url_for("knowledge_base"))

    # 1. Delete all documents from ChromaDB one by one (safer than dropping
    #    the whole collection while the client may hold a handle to it)
    try:
        sources = list_sources(project_id)
        for filename in list(sources.keys()):
            try:
                delete_document(project_id, filename)
            except Exception as exc:
                logging.warning("Could not remove '%s' from ChromaDB: %s", filename, exc)
        # Now try dropping the collection entirely
        from rag import _get_client
        import re as _re
        sanitised = _re.sub(r"[^a-zA-Z0-9_-]", "_", project_id).strip("_-") or "default"
        collection_name = "project-" + sanitised[:57]  # keep under 63 chars total
        try:
            _get_client().delete_collection(collection_name)
        except Exception:
            pass
    except Exception as exc:
        logging.warning("ChromaDB cleanup error for '%s': %s", project_id, exc)

    # 2. Remove files from disk one by one, then remove the folder
    project_dir = UPLOAD_DIR / project_id
    errors = []
    if project_dir.exists():
        # Delete each file individually first
        for f in project_dir.iterdir():
            try:
                f.unlink()
            except PermissionError:
                # On Windows, try clearing the read-only bit then retry
                try:
                    f.chmod(stat.S_IWRITE)
                    f.unlink()
                except Exception as exc:
                    errors.append(f.name)
                    logging.warning("Could not delete file '%s': %s", f, exc)

        # Remove the now-empty folder
        if not errors:
            try:
                project_dir.rmdir()
            except Exception:
                # Fall back to rmtree with an error handler
                def _force_remove(func, path, _):
                    try:
                        import os
                        os.chmod(path, stat.S_IWRITE)
                        func(path)
                    except Exception:
                        pass
                shutil.rmtree(str(project_dir), onerror=_force_remove)

    if errors:
        flash(
            f"Project '{project_id}' removed from the knowledge base, but "
            f"{len(errors)} file(s) could not be deleted from disk "
            f"(they may be open in another program): {', '.join(errors)}",
            "info",
        )
    else:
        flash(f"Project '{project_id}' and all its documents have been deleted.", "success")

    # Remove persisted analysis report (best-effort)
    delete_report(project_id)

    return redirect(url_for("knowledge_base"))


@app.route("/search", methods=["GET", "POST"])
def search_kb():
    import logging
    projects   = get_all_projects()
    results    = []
    query      = ""
    k          = 10
    # Seed project selector from URL param on GET; overridden by form on POST
    project_id = request.args.get("project_id", "").strip() or \
                 (projects[0]["id"] if projects else "")

    if request.method == "POST":
        project_id = request.form.get("project_id", "").strip()
        query      = request.form.get("query",      "").strip()

        # Guard against non-integer k values
        try:
            k = int(request.form.get("k", 5))
            if k < 1:
                k = 1
        except (ValueError, TypeError):
            k = 5

        if not project_id:
            flash("Please select a project.", "error")
        elif not query:
            flash("Please enter a search query.", "error")
        else:
            # Check the project has indexed documents before hitting the model
            stats = get_project_stats(project_id)
            if stats["chunk_count"] == 0:
                flash(
                    f"No documents are indexed for '{project_id}' yet. "
                    "Go to Upload and add some documents first.",
                    "error",
                )
            else:
                try:
                    hits = search(project_id, query, k=k)
                    results = [
                        {
                            "source":      h.source,
                            "chunk_index": h.chunk_index,
                            "score":       h.score,
                            "text":        h.text,
                        }
                        for h in hits
                    ]
                    if not results:
                        flash(
                            "No matching chunks found for that query. "
                            "Try different keywords.",
                            "info",
                        )
                except Exception as exc:
                    logging.exception("Search failed for project '%s': %s", project_id, exc)
                    flash(
                        f"Search error: {exc}. "
                        "Check that the embedding model is installed and the vector store is accessible.",
                        "error",
                    )

    return render_template(
        "search.html",
        projects=projects,
        selected_project=project_id,
        query=query,
        results=results,
        k=k,
    )


@app.route("/analyse", methods=["GET", "POST"])
def analyse():
    """Run the three-agent pipeline on a selected project and show the report."""
    import logging
    projects   = get_all_projects()
    report     = None
    project_id = request.args.get("project_id", "").strip()

    if request.method == "POST":
        project_id = request.form.get("project_id", "").strip()

        if not project_id:
            flash("Please select a project to analyse.", "error")
            return redirect(url_for("analyse"))

        stats = get_project_stats(project_id)
        if stats["chunk_count"] == 0:
            flash(
                f"No documents are indexed for '{project_id}'. "
                "Upload documents first.",
                "error",
            )
            return redirect(url_for("analyse"))

        try:
            from agents.orchestrator import run as run_analysis
            report = run_analysis(project_id)
            # Persist the report so the dashboard can show insights without
            # re-running the analysis on every page load.
            save_report(project_id, report)
        except Exception as exc:
            logging.exception("Orchestrator failed for project '%s': %s", project_id, exc)
            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return jsonify({"error": str(exc)}), 500
            flash(f"Analysis failed: {exc}", "error")

    # AJAX request from the single-page app → return JSON report
    if request.method == "POST" and request.headers.get("X-Requested-With") == "XMLHttpRequest":
        if report:
            return jsonify({"report": report, "project_id": project_id})
        return jsonify({"error": "No report generated"}), 400

    return render_template(
        "analysis.html",
        projects=projects,
        selected_project=project_id,
        report=report,
    )


# ---------------------------------------------------------------------------
# Milestone 3 — Conversational Q&A Assistant
# ---------------------------------------------------------------------------

@app.route("/qa")
def qa_page():
    """Render the Q&A chat page."""
    projects = get_all_projects()
    project_id = request.args.get("project_id", "").strip() or \
                 (projects[0]["id"] if projects else "")
    return render_template("qa.html", projects=projects, selected_project=project_id)


@app.route("/ask", methods=["POST"])
def ask():
    """Answer a project question using RAG + LLM. Returns JSON."""
    import logging
    from flask import jsonify

    project_id = request.form.get("project_id", "").strip()
    question   = request.form.get("question", "").strip()

    if not project_id:
        return jsonify({"error": "Please select a project."}), 400
    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    stats = get_project_stats(project_id)
    if stats["chunk_count"] == 0:
        return jsonify({
            "error": f"No documents indexed for '{project_id}'. Upload documents first."
        }), 400

    try:
        from agents.qa_agent import answer as qa_answer
        from agents.llm_client import clear_context_cache
        # Clear context cache so the Q&A gets fresh retrieval
        clear_context_cache(project_id)
        result = qa_answer(project_id, question)
        return jsonify(result)
    except Exception as exc:
        logging.exception("Q&A failed for project '%s': %s", project_id, exc)
        return jsonify({"error": f"Q&A error: {exc}"}), 500


# ---------------------------------------------------------------------------
# Milestone 4 — Single-Page Application & PDF Report
# ---------------------------------------------------------------------------

@app.route("/app")
def spa():
    """Single-page application — all sections on one page."""
    projects = get_all_projects()
    selected_project = request.args.get("project_id", "").strip() or \
                       (projects[0]["id"] if projects else "")
    # Load last saved report if available
    from report_store import load_report
    report = None
    if selected_project:
        report = load_report(selected_project)
    sources = {}
    if selected_project:
        sources = list_sources(selected_project)
    return render_template(
        "app_v2.html",
        projects=projects,
        selected_project=selected_project,
        report=report,
        sources=sources,
        allowed=sorted(ALLOWED_EXTENSIONS),
    )


@app.route("/api/sources")
def api_sources():
    """Return indexed file list for a project as JSON (for AJAX KB loading)."""
    project_id = request.args.get("project_id", "").strip()
    if not project_id:
        return jsonify({"sources": {}, "error": "No project specified"}), 400
    sources = list_sources(project_id)
    stats = get_project_stats(project_id)
    return jsonify({"sources": sources, "stats": stats})


@app.route("/api/projects")
def api_projects():
    """Return all projects with their summaries as JSON."""
    projects = get_all_projects()
    return jsonify({"projects": projects})


@app.route("/api/report/<project_id>")
def api_report(project_id):
    """Return the saved analysis report for a project as JSON."""
    from report_store import load_report
    report = load_report(project_id)
    if not report:
        return jsonify({"error": "No report found"}), 404
    return jsonify({"report": report, "project_id": project_id})


@app.route("/report/pdf")
def report_pdf():
    """Generate a downloadable PDF project report using existing analysis data."""
    import io as _io
    import logging as _log
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib.colors import HexColor
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
        )
        from reportlab.lib.enums import TA_CENTER
    except ImportError:
        return jsonify({"error": "reportlab not installed. Run: pip install reportlab"}), 500

    project_id = request.args.get("project_id", "").strip()
    if not project_id:
        return jsonify({"error": "No project specified"}), 400

    from report_store import load_report
    report = load_report(project_id)
    if not report:
        return jsonify({"error": "No analysis report found. Run analysis first."}), 404

    buf = _io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm,
                            leftMargin=2*cm, rightMargin=2*cm)
    styles = getSampleStyleSheet()
    story = []

    # ── Cover page ──
    story.append(Spacer(1, 6*cm))
    story.append(Paragraph("Project Intelligence Report", ParagraphStyle(
        'CoverTitle', parent=styles['Title'], fontSize=28, alignment=TA_CENTER,
        textColor=HexColor('#1e3a5f'), spaceAfter=12)))
    project_name = project_id.replace("-", " ").title()
    story.append(Paragraph(project_name, ParagraphStyle(
        'CoverSubtitle', parent=styles['Normal'], fontSize=18, alignment=TA_CENTER,
        textColor=HexColor('#3d6491'), spaceAfter=30)))
    gen_at = report.get("generated_at", "N/A")
    story.append(Paragraph(f"Generated: {gen_at}", ParagraphStyle(
        'CoverMeta', parent=styles['Normal'], fontSize=11, alignment=TA_CENTER,
        textColor=HexColor('#6b7280'))))
    hs = report.get("health_score", {})
    if hs.get("overall") is not None:
        story.append(Spacer(1, 20))
        story.append(Paragraph(f"Health Score: {hs.get('overall', 0)}/100  (Grade {hs.get('grade', 'F')})",
            ParagraphStyle('CoverScore', parent=styles['Normal'], fontSize=14,
            alignment=TA_CENTER, textColor=HexColor('#1e4a31' if hs.get('overall', 0) >= 70 else '#b85c5c'))))
    story.append(PageBreak())

    # ── Helper for section headings ──
    def heading(text, level=1):
        sz = 16 if level == 1 else 13
        clr = HexColor('#1e3a5f') if level == 1 else HexColor('#2c4a6e')
        parent_style = styles['Heading1'] if level == 1 else styles['Heading2']
        story.append(Paragraph(text, ParagraphStyle(
            f'CustomH{level}_{len(story)}', parent=parent_style, fontSize=sz,
            textColor=clr, spaceBefore=16, spaceAfter=8)))

    def para(text):
        story.append(Paragraph(text, styles['Normal']))
        story.append(Spacer(1, 6))

    def make_table(data, col_widths=None):
        t = Table(data, colWidths=col_widths)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#eaf1fb')),
            ('TEXTCOLOR', (0, 0), (-1, 0), HexColor('#2c4a6e')),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#dde3ed')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor('#ffffff'), HexColor('#f7faff')]),
        ]))
        return t

    # ── 1. Project Overview ──
    heading("1. Project Overview")
    para(f"<b>Project:</b> {project_name}")
    para(f"<b>Project ID:</b> {project_id}")
    para(f"<b>Generated:</b> {gen_at}")
    para(f"<b>Duration:</b> {report.get('duration_seconds', 0)}s")
    para(f"<b>Status:</b> {report.get('status', 'unknown')}")

    # ── 2. Project Objectives ──
    scope = report.get("scope", {})
    heading("2. Project Objectives")
    goals = scope.get("project_goals", [])
    if goals:
        for g in goals:
            goal_text = g.get("goal", "") if isinstance(g, dict) else str(g)
            src = g.get("source", "") if isinstance(g, dict) else ""
            para(f"• {goal_text}" + (f" <i>({src})</i>" if src else ""))
    else:
        para("<i>No project goals were extracted from the documents.</i>")

    # ── 3. Scope and Deliverables ──
    heading("3. Scope and Deliverables")
    scope_items = scope.get("scope", [])
    if scope_items:
        heading("In-Scope Items", 2)
        for s in scope_items:
            item = s.get("item", "") if isinstance(s, dict) else str(s)
            para(f"• {item}")

    oos_items = scope.get("out_of_scope", [])
    if oos_items:
        heading("Out-of-Scope Items", 2)
        for s in oos_items:
            item = s.get("item", "") if isinstance(s, dict) else str(s)
            para(f"• {item}")

    deliverables = scope.get("deliverables", [])
    if deliverables:
        heading("Deliverables", 2)
        data = [["Name", "Description", "Deadline", "Owner", "Source"]]
        for d in deliverables:
            data.append([
                d.get("name", ""),
                d.get("description", "") or "—",
                str(d.get("deadline") or "—"),
                d.get("owner") or "—",
                d.get("source", ""),
            ])
        story.append(make_table(data, [3*cm, 5*cm, 2.5*cm, 2.5*cm, 3*cm]))
        story.append(Spacer(1, 8))

    # ── 4. Milestones ──
    milestones = scope.get("milestones", [])
    if milestones:
        heading("4. Project Milestones")
        data = [["Milestone", "Date", "Description", "Source"]]
        for m in milestones:
            data.append([
                m.get("name", ""),
                str(m.get("date") or "—"),
                m.get("description", "") or "—",
                m.get("source", ""),
            ])
        story.append(make_table(data, [4*cm, 2.5*cm, 5*cm, 3*cm]))
        story.append(Spacer(1, 8))

    # ── 5. Risk Register ──
    risks = report.get("risks", {})
    risk_list = risks.get("risks", [])
    if risk_list:
        heading("5. Risk Register")
        data = [["ID", "Title", "Severity", "Impact", "Mitigation", "Status"]]
        for r in risk_list:
            data.append([
                r.get("risk_id", ""),
                r.get("title", "")[:40],
                r.get("severity", "Medium"),
                (r.get("potential_impact") or r.get("description", ""))[:50],
                (r.get("mitigation") or "—")[:50],
                r.get("status", "Unknown"),
            ])
        story.append(make_table(data, [1.5*cm, 3.5*cm, 2*cm, 4*cm, 4*cm, 2.5*cm]))
        story.append(Spacer(1, 8))

    # ── 6. Blockers and Action Items ──
    blockers = report.get("blockers", {})
    blocker_list = blockers.get("blockers", [])
    if blocker_list:
        heading("6. Blockers")
        data = [["ID", "Description", "Status", "Impact", "Owner"]]
        for b in blocker_list:
            data.append([
                b.get("blocker_id", ""),
                b.get("description", "")[:40],
                b.get("status", "Unknown"),
                (b.get("impact") or "—")[:40],
                b.get("owner") or "—",
            ])
        story.append(make_table(data, [1.5*cm, 5*cm, 2*cm, 4*cm, 3*cm]))
        story.append(Spacer(1, 8))

    action_items = blockers.get("action_items", [])
    if action_items:
        heading("Action Items", 2)
        data = [["ID", "Description", "Owner", "Due Date", "Priority", "Status"]]
        for a in action_items:
            data.append([
                a.get("action_id", ""),
                a.get("description", "")[:35],
                a.get("owner") or "—",
                str(a.get("due_date") or "—"),
                a.get("priority", "—"),
                a.get("status", "—"),
            ])
        story.append(make_table(data, [1.5*cm, 4.5*cm, 2.5*cm, 2.5*cm, 2*cm, 2.5*cm]))
        story.append(Spacer(1, 8))

    # ── 7. Timeline and Delivery Forecast ──
    fc = risks.get("delivery_forecast", {})
    if fc:
        heading("7. Timeline and Delivery Forecast")
        para(f"<b>Status:</b> {fc.get('status', 'Insufficient data')}")
        if fc.get("estimated_completion_date"):
            para(f"<b>Estimated Completion:</b> {fc['estimated_completion_date']}")
        if fc.get("confidence"):
            para(f"<b>Confidence:</b> {fc['confidence']}")
        if fc.get("forecast_method"):
            para(f"<b>Method:</b> {fc['forecast_method']}")
        if fc.get("delayed_milestones"):
            para(f"<b>Delayed Milestones:</b> {', '.join(fc['delayed_milestones'])}")
        if fc.get("assumptions"):
            para("<b>Assumptions:</b>")
            for a in fc["assumptions"]:
                para(f"  • {a}")

    # ── 8. Project Health Score ──
    if hs.get("overall") is not None:
        heading("8. Project Health Score")
        para(f"<b>Overall Score:</b> {hs.get('overall', 0)}/100  (Grade {hs.get('grade', 'F')})")
        if hs.get("summary"):
            para(f"<b>Summary:</b> {hs['summary']}")
        dims = hs.get("dimensions", [])
        if dims:
            data = [["Dimension", "Score", "Weight", "Details"]]
            for d in dims:
                data.append([
                    d.get("name", ""),
                    f"{d.get('score', 0)}/100",
                    f"{int(d.get('weight', 0)*100)}%",
                    (d.get("details", "") or "")[:60],
                ])
            story.append(make_table(data, [3.5*cm, 2*cm, 2*cm, 8*cm]))

    # ── 9. Missing Information ──
    warnings = report.get("warnings", [])
    if warnings:
        heading("9. Missing Information and Recommendations")
        for w in warnings:
            para(f"• {w}")
    else:
        heading("9. Missing Information and Recommendations")
        para("<i>No missing information flagged.</i>")

    # ── 10. User Stories ──
    docs = report.get("docs", {})
    user_stories = docs.get("user_stories", [])
    if user_stories:
        heading("10. Generated User Stories")
        data = [["ID", "Role", "Want", "Benefit"]]
        for us in user_stories:
            data.append([
                us.get("id", ""),
                us.get("role", ""),
                us.get("want", "")[:35],
                us.get("benefit", "")[:40],
            ])
        story.append(make_table(data, [1.5*cm, 2.5*cm, 5*cm, 6.5*cm]))
        story.append(Spacer(1, 8))

    # ── 11. Conclusion ──
    heading("11. Project Conclusion")
    overall = hs.get("overall", 0)
    if overall >= 80:
        para("The project is in good health with well-documented scope, manageable risks, and few blockers. "
             "Continue monitoring and maintain the current pace.")
    elif overall >= 60:
        para("The project has moderate concerns. Some scope items, risks, or blockers need attention. "
             "Review the highlighted areas and take corrective action.")
    elif overall >= 40:
        para("The project is at risk. Several issues need immediate attention — including scope gaps, "
             "open blockers, or high-severity risks. Prioritise resolving these before proceeding.")
    else:
        para("The project is in critical condition. Immediate action is required on multiple fronts. "
             "Review all blockers, risks, and missing information before continuing.")

    # ── Footer with page numbers ──
    def _footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(HexColor('#9ca3af'))
        canvas.drawRightString(A4[0] - 2*cm, 1*cm, f"Page {doc_.page}")
        canvas.drawString(2*cm, 1*cm, f"AI Project Intelligence Report — {project_name}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    buf.seek(0)
    from flask import Response
    return Response(
        buf.getvalue(),
        mimetype="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=project_report_{project_id}.pdf"},
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Safe diagnostic — never prints the key value
    from agents.llm_client import check_api_key
    diag = check_api_key()
    if diag["configured"]:
        print(f"[startup] GROQ_API_KEY configured (source: {diag['source']})")
    else:
        print(f"[startup] WARNING — GROQ_API_KEY not set. {diag['hint']}")

    # Pre-load the embedding model before the first request so uploads
    # don't pay the cold-start penalty (~2-3 s on CPU).
    print("[startup] Loading embedding model…")
    get_model()
    print("[startup] Model ready.")
    app.run(debug=True, port=5000)
