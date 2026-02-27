#!/usr/bin/env python3
"""Simple web viewer for PC-Eval benchmark results.

This creates a local web server to view benchmark results with charts and details.

Usage:
    python view_results.py [results_dir]

    # View latest results
    python view_results.py

    # View specific run
    python view_results.py pc_eval_live_results/run_20240225_143022_claude-sonnet-45
"""

import argparse
import json
import os
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import webbrowser
from typing import Dict, List

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>PC-Eval Benchmark Results</title>
    <meta charset="utf-8">
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 30px;
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        .stat-card {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            border: 1px solid #e9ecef;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #495057;
        }
        .stat-label {
            color: #6c757d;
            margin-top: 5px;
        }
        .success { color: #28a745; }
        .failed { color: #dc3545; }
        .timeout { color: #ffc107; }
        .tasks-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 30px;
        }
        .tasks-table th {
            background: #f8f9fa;
            padding: 12px;
            text-align: left;
            border-bottom: 2px solid #dee2e6;
        }
        .tasks-table td {
            padding: 12px;
            border-bottom: 1px solid #dee2e6;
        }
        .tasks-table tr:hover {
            background: #f8f9fa;
        }
        .status-badge {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: bold;
        }
        .status-success {
            background: #d4edda;
            color: #155724;
        }
        .status-failed {
            background: #f8d7da;
            color: #721c24;
        }
        .status-timeout {
            background: #fff3cd;
            color: #856404;
        }
        .progress-bar {
            width: 100%;
            height: 30px;
            background: #e9ecef;
            border-radius: 15px;
            overflow: hidden;
            margin: 20px 0;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #28a745 0%, #20c997 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }
        .chart-container {
            margin: 30px 0;
        }
        .bar {
            fill: #667eea;
        }
        .bar:hover {
            fill: #764ba2;
        }
        .task-details {
            font-size: 0.9em;
            color: #6c757d;
            max-width: 500px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 PC-Eval Benchmark Results</h1>
        <div class="model-info">
            <strong>Model:</strong> {model_name}<br>
            <strong>Run Time:</strong> {timestamp}<br>
            <strong>Tasks:</strong> {total_tasks}
        </div>

        <div class="progress-bar">
            <div class="progress-fill" style="width: {success_rate}%">
                Success Rate: {success_rate}%
            </div>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value success">{successful}</div>
                <div class="stat-label">Successful</div>
            </div>
            <div class="stat-card">
                <div class="stat-value failed">{failed}</div>
                <div class="stat-label">Failed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value timeout">{timeout}</div>
                <div class="stat-label">Timeout</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{avg_duration}s</div>
                <div class="stat-label">Avg Duration</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{avg_steps}</div>
                <div class="stat-label">Avg Steps</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{total_duration}m</div>
                <div class="stat-label">Total Time</div>
            </div>
        </div>

        <h2>Task Details</h2>
        <table class="tasks-table">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Description</th>
                    <th>Status</th>
                    <th>Duration</th>
                    <th>Steps</th>
                </tr>
            </thead>
            <tbody>
                {task_rows}
            </tbody>
        </table>

        <div class="chart-container">
            <h3>Task Duration Distribution</h3>
            <svg id="duration-chart" width="100%" height="300"></svg>
        </div>
    </div>

    <script>
        // Add interactive features if needed
        document.querySelectorAll('.tasks-table tr').forEach(row => {{
            row.style.cursor = 'pointer';
            row.onclick = function() {{
                // Could expand to show more details
            }};
        }});
    </script>
</body>
</html>
"""


def generate_html_report(results_dir: Path) -> str:
    """Generate HTML report from results directory."""

    # Load summary.json
    summary_file = results_dir / "summary.json"
    if not summary_file.exists():
        return "<html><body><h1>No summary.json found in this directory</h1></body></html>"

    with open(summary_file) as f:
        data = json.load(f)

    model = data.get("model", {})
    results = data.get("results", [])
    timestamp = data.get("timestamp", "Unknown")

    # Calculate statistics
    total_tasks = len(results)
    successful = sum(1 for r in results if r["status"] == "SUCCESS")
    failed = sum(1 for r in results if r["status"] == "FAILED")
    timeout = sum(1 for r in results if r["status"] == "TIMEOUT")

    success_rate = round((successful / total_tasks * 100) if total_tasks > 0 else 0, 1)

    durations = [r["duration_seconds"] for r in results]
    avg_duration = round(sum(durations) / len(durations), 1) if durations else 0
    total_duration = round(sum(durations) / 60, 1)  # in minutes

    steps = [r["total_steps"] for r in results]
    avg_steps = round(sum(steps) / len(steps), 1) if steps else 0

    # Generate task rows
    task_rows = []
    for r in results:
        status = r["status"]
        status_class = "success" if status == "SUCCESS" else "failed" if status == "FAILED" else "timeout"
        desc = r["description"]
        if len(desc) > 80:
            desc = desc[:77] + "..."

        task_rows.append(f"""
            <tr>
                <td>{r['task_num']}</td>
                <td class="task-details">{desc}</td>
                <td><span class="status-badge status-{status_class}">{status}</span></td>
                <td>{r['duration_seconds']}s</td>
                <td>{r['total_steps']}</td>
            </tr>
        """)

    # Fill template
    html = HTML_TEMPLATE.format(
        model_name=model.get("title", model.get("name", "Unknown")),
        timestamp=timestamp,
        total_tasks=total_tasks,
        success_rate=success_rate,
        successful=successful,
        failed=failed,
        timeout=timeout,
        avg_duration=avg_duration,
        avg_steps=avg_steps,
        total_duration=total_duration,
        task_rows="".join(task_rows)
    )

    return html


def main():
    parser = argparse.ArgumentParser(description="View PC-Eval benchmark results")
    parser.add_argument("results_dir", nargs="?", help="Results directory to view")
    parser.add_argument("--port", type=int, default=8080, help="Port for web server")
    args = parser.parse_args()

    # Find results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        # Find latest results
        base_dirs = [
            Path("pc_eval_live_results"),
            Path("pc_eval_results")
        ]

        latest_run = None
        latest_time = None

        for base_dir in base_dirs:
            if base_dir.exists():
                for run_dir in base_dir.glob("run_*"):
                    if run_dir.is_dir():
                        mtime = run_dir.stat().st_mtime
                        if latest_time is None or mtime > latest_time:
                            latest_time = mtime
                            latest_run = run_dir

        if latest_run:
            results_dir = latest_run
        else:
            print("No results found. Run a benchmark first!")
            return

    print(f"📊 Viewing results from: {results_dir}")

    # Generate HTML
    html = generate_html_report(results_dir)

    # Save HTML file
    html_file = results_dir / "report.html"
    with open(html_file, "w") as f:
        f.write(html)

    print(f"📄 Report generated: {html_file}")

    # Open in browser
    url = f"file://{html_file.absolute()}"
    print(f"🌐 Opening in browser: {url}")
    webbrowser.open(url)


if __name__ == "__main__":
    main()