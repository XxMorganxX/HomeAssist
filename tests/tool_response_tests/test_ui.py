"""
Simple Tkinter UI for viewing and running MCP tool response tests.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import json
import asyncio
import threading
from pathlib import Path
from datetime import datetime


class ToolTestUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MCP Tool Response Tests")
        self.root.geometry("1200x800")
        
        # Paths
        self.tests_dir = Path(__file__).parent
        self.input_file = self.tests_dir / "input.json"
        self.output_file = self.tests_dir / "output.json"
        
        # Data
        self.input_data = []
        self.output_data = {}
        
        self._create_ui()
        self._load_data()
    
    def _create_ui(self):
        """Create the UI layout."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Top bar with buttons and status
        top_frame = ttk.Frame(main_frame)
        top_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        
        self.run_btn = ttk.Button(top_frame, text="▶ Run Tests", command=self._run_tests)
        self.run_btn.pack(side="left", padx=(0, 10))
        
        self.refresh_btn = ttk.Button(top_frame, text="🔄 Refresh", command=self._load_data)
        self.refresh_btn.pack(side="left", padx=(0, 10))
        
        self.status_label = ttk.Label(top_frame, text="Ready", foreground="gray")
        self.status_label.pack(side="left", padx=(10, 0))
        
        # Stats frame
        self.stats_frame = ttk.LabelFrame(top_frame, text="Stats", padding="5")
        self.stats_frame.pack(side="right")
        
        self.stats_label = ttk.Label(self.stats_frame, text="No results loaded")
        self.stats_label.pack()
        
        # Left panel - Input prompts
        left_frame = ttk.LabelFrame(main_frame, text="Input Prompts", padding="5")
        left_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Input tree view
        self.input_tree = ttk.Treeview(left_frame, columns=("prompt",), show="tree headings")
        self.input_tree.heading("#0", text="Category")
        self.input_tree.heading("prompt", text="Prompt")
        self.input_tree.column("#0", width=120)
        self.input_tree.column("prompt", width=300)
        self.input_tree.pack(fill="both", expand=True, side="left")
        
        input_scroll = ttk.Scrollbar(left_frame, orient="vertical", command=self.input_tree.yview)
        input_scroll.pack(side="right", fill="y")
        self.input_tree.configure(yscrollcommand=input_scroll.set)
        
        self.input_tree.bind("<<TreeviewSelect>>", self._on_input_select)
        
        # Right panel - Output details
        right_frame = ttk.LabelFrame(main_frame, text="Test Result", padding="5")
        right_frame.grid(row=1, column=1, sticky="nsew", padx=(5, 0))
        main_frame.columnconfigure(1, weight=2)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill="both", expand=True)
        
        # Response tab
        response_frame = ttk.Frame(self.notebook)
        self.notebook.add(response_frame, text="Response")
        
        self.response_text = scrolledtext.ScrolledText(response_frame, wrap="word", height=10)
        self.response_text.pack(fill="both", expand=True)
        
        # Tool Signal Flow tab
        flow_frame = ttk.Frame(self.notebook)
        self.notebook.add(flow_frame, text="Tool Signal Flow")
        
        self.flow_text = scrolledtext.ScrolledText(flow_frame, wrap="word", height=10, font=("Courier", 11))
        self.flow_text.pack(fill="both", expand=True)
        
        # Tools Used tab
        tools_frame = ttk.Frame(self.notebook)
        self.notebook.add(tools_frame, text="Tools Used")
        
        self.tools_text = scrolledtext.ScrolledText(tools_frame, wrap="word", height=10, font=("Courier", 10))
        self.tools_text.pack(fill="both", expand=True)
        
        # Configure color tags for tools text
        self.tools_text.tag_config("header", foreground="#2563eb", font=("Courier", 11, "bold"))
        self.tools_text.tag_config("tool_name", foreground="#7c3aed", font=("Courier", 10, "bold"))
        self.tools_text.tag_config("section_label", foreground="#059669", font=("Courier", 10, "bold"))
        self.tools_text.tag_config("timing", foreground="#dc2626", font=("Courier", 10))
        self.tools_text.tag_config("arguments", foreground="#0891b2")
        self.tools_text.tag_config("result", foreground="#65a30d")
        self.tools_text.tag_config("final_response", foreground="#ea580c", font=("Courier", 10, "bold"))
        self.tools_text.tag_config("separator", foreground="#6b7280")
        
        # Console/Log tab
        console_frame = ttk.Frame(self.notebook)
        self.notebook.add(console_frame, text="Console")
        
        self.console_text = scrolledtext.ScrolledText(console_frame, wrap="word", height=10, font=("Courier", 10))
        self.console_text.pack(fill="both", expand=True)
    
    def _load_data(self):
        """Load input and output data."""
        # Load inputs
        try:
            with open(self.input_file, "r") as f:
                self.input_data = json.load(f)
            self._populate_input_tree()
        except FileNotFoundError:
            messagebox.showerror("Error", f"Input file not found: {self.input_file}")
        except json.JSONDecodeError as e:
            messagebox.showerror("Error", f"Invalid JSON in input file: {e}")
        
        # Load outputs
        try:
            with open(self.output_file, "r") as f:
                self.output_data = json.load(f)
            self._update_stats()
        except FileNotFoundError:
            self.output_data = {}
            self.stats_label.config(text="No results yet")
        except json.JSONDecodeError as e:
            self.output_data = {}
            self.stats_label.config(text="Invalid output file")
        
        self.status_label.config(text="Data loaded", foreground="green")
    
    def _populate_input_tree(self):
        """Populate the input tree view."""
        # Clear existing items
        for item in self.input_tree.get_children():
            self.input_tree.delete(item)
        
        # Add categories and prompts
        for category_obj in self.input_data:
            for category, prompts in category_obj.items():
                # Add category as parent
                cat_id = self.input_tree.insert("", "end", text=category, open=True)
                
                # Add prompts as children
                for prompt in prompts:
                    self.input_tree.insert(cat_id, "end", values=(prompt,))
    
    def _update_stats(self):
        """Update the stats display."""
        if not self.output_data:
            self.stats_label.config(text="No results")
            return
        
        stats = self.output_data.get("tool_signal_stats", {})
        total = stats.get("total_tests", 0)
        signal = stats.get("tool_signal_detected", 0)
        orch = stats.get("orchestration_called", 0)
        tools = stats.get("tools_used", 0)
        
        signal_pct = f"{100*signal//max(1,total)}%" if total else "N/A"
        
        # Get timing summary
        timing = self.output_data.get("timing_summary", {})
        avg_time = timing.get("total_time", {}).get("avg")
        avg_str = f" | Avg: {avg_time}ms" if avg_time else ""
        
        text = f"Tests: {total} | Signal: {signal} ({signal_pct}) | Tools: {tools}{avg_str}"
        self.stats_label.config(text=text)
    
    def _on_input_select(self, event):
        """Handle input selection."""
        selection = self.input_tree.selection()
        if not selection:
            return
        
        item = selection[0]
        values = self.input_tree.item(item, "values")
        
        if not values:
            # Category selected, not a prompt
            return
        
        prompt = values[0]
        self._show_result(prompt)
    
    def _show_result(self, prompt):
        """Show the result for a given prompt."""
        # Find the result in output data
        result = None
        for r in self.output_data.get("results", []):
            if r.get("prompt") == prompt:
                result = r
                break
        
        if not result:
            self.response_text.delete("1.0", "end")
            self.response_text.insert("1.0", "No result found for this prompt.\n\nRun the tests to generate results.")
            self.flow_text.delete("1.0", "end")
            self.tools_text.delete("1.0", "end")
            return
        
        # Response tab
        self.response_text.delete("1.0", "end")
        self.response_text.insert("1.0", f"Prompt: {prompt}\n\n")
        
        # Show timing at the top
        timing = result.get("timing", {})
        total_ms = timing.get("total_time_ms")
        tool_ms = timing.get("tool_execution_time_ms")
        if total_ms:
            self.response_text.insert("end", f"⏱️ Total: {total_ms}ms ({total_ms/1000:.2f}s)")
            if tool_ms:
                self.response_text.insert("end", f"  |  Tool exec: {tool_ms}ms")
            self.response_text.insert("end", "\n\n")
        
        self.response_text.insert("end", "─" * 50 + "\n\n")
        self.response_text.insert("end", result.get("response", "(no response)"))
        
        # Tool Signal Flow tab
        self.flow_text.delete("1.0", "end")
        flow = result.get("tool_signal_flow", {})
        timing = result.get("timing", {})
        
        # Format timing values
        total_ms = timing.get("total_time_ms")
        tool_ms = timing.get("tool_execution_time_ms") or flow.get("tool_execution_time_ms")
        total_str = f"{total_ms}ms" if total_ms else "N/A"
        tool_str = f"{tool_ms}ms" if tool_ms else "N/A"
        
        flow_lines = [
            f"┌─────────────────────────────────────────┐",
            f"│         TOOL SIGNAL FLOW                │",
            f"├─────────────────────────────────────────┤",
            f"│ Signal Detected:     {str(flow.get('tool_signal_detected', 'N/A')):>17} │",
            f"│ Realtime Output:     {str(flow.get('realtime_raw_output', 'N/A'))[:17]:>17} │",
            f"│ Orchestration:       {str(flow.get('orchestration_called', 'N/A')):>17} │",
            f"│ Orchestration Model: {str(flow.get('orchestration_model', 'N/A')):>17} │",
            f"│ Realtime Compose:    {str(flow.get('realtime_compose_called', 'N/A')):>17} │",
            f"├─────────────────────────────────────────┤",
            f"│             TIMING                      │",
            f"├─────────────────────────────────────────┤",
            f"│ Total Time:          {total_str:>17} │",
            f"│ Tool Execution:      {tool_str:>17} │",
            f"└─────────────────────────────────────────┘",
        ]
        self.flow_text.insert("1.0", "\n".join(flow_lines))
        
        # Add visual flow diagram
        if flow.get("tool_signal_detected"):
            diagram = """
            
Flow Diagram:
─────────────

User Request
     │
     ▼
┌─────────────┐
│  Realtime   │ ─── outputs "TOOL"
└─────────────┘
     │
     ▼
┌─────────────┐
│ gpt-4o-mini │ ─── orchestrates tools
└─────────────┘
     │
     ▼
┌─────────────┐
│  Tool(s)    │ ─── executes
└─────────────┘
     │
     ▼
┌─────────────┐
│  Realtime   │ ─── composes response
└─────────────┘
     │
     ▼
Final Response
"""
            self.flow_text.insert("end", diagram)
        
        # Tools Used tab
        self.tools_text.delete("1.0", "end")
        tools_used = result.get("tools_used", [])
        
        # Get timing info
        timing = result.get("timing", {})
        total_ms = timing.get("total_time_ms")
        tool_exec_ms = timing.get("tool_execution_time_ms")
        
        # Header with timing
        self._insert_colored(self.tools_text, "TOOLS EXECUTION SUMMARY\n", "header")
        self._insert_colored(self.tools_text, "═" * 60 + "\n", "separator")
        
        if total_ms:
            self._insert_colored(self.tools_text, "⏱️ Timing:\n", "section_label")
            self._insert_colored(self.tools_text, f"  Total time: {total_ms}ms ({total_ms/1000:.2f}s)\n", "timing")
            if tool_exec_ms:
                self._insert_colored(self.tools_text, f"  Tool execution: {tool_exec_ms}ms\n", "timing")
            self.tools_text.insert("end", "\n")
        
        if not tools_used:
            self.tools_text.insert("end", "No tools were used for this prompt.")
        else:
            self._insert_colored(self.tools_text, f"📦 Tools Called: {len(tools_used)}\n\n", "section_label")
            
            for i, tool in enumerate(tools_used, 1):
                self._insert_colored(self.tools_text, "─" * 60 + "\n", "separator")
                self._insert_colored(self.tools_text, f"TOOL {i}: ", "tool_name")
                self._insert_colored(self.tools_text, f"{tool.get('name', 'Unknown')}\n", "tool_name")
                self._insert_colored(self.tools_text, "─" * 60 + "\n", "separator")
                self.tools_text.insert("end", "\n")
                
                # Arguments
                self._insert_colored(self.tools_text, "📥 Arguments:\n", "section_label")
                args = tool.get("arguments", {})
                args_str = json.dumps(args, indent=2)
                self._insert_colored(self.tools_text, args_str + "\n\n", "arguments")
                
                # Result
                self._insert_colored(self.tools_text, "📤 Result:\n", "section_label")
                result_data = tool.get("result")
                if result_data:
                    try:
                        parsed = json.loads(result_data) if isinstance(result_data, str) else result_data
                        result_str = json.dumps(parsed, indent=2)[:2000]
                        self._insert_colored(self.tools_text, result_str, "result")
                    except (json.JSONDecodeError, TypeError):
                        self._insert_colored(self.tools_text, str(result_data)[:2000], "result")
                else:
                    self.tools_text.insert("end", "(no result)")
                
                self.tools_text.insert("end", "\n\n")
        
        # Final Response section
        self._insert_colored(self.tools_text, "\n" + "═" * 60 + "\n", "separator")
        self._insert_colored(self.tools_text, "💬 FINAL RESPONSE\n", "final_response")
        self._insert_colored(self.tools_text, "═" * 60 + "\n\n", "separator")
        
        final_response = result.get("response", "(no response)")
        self._insert_colored(self.tools_text, final_response, "final_response")
    
    def _insert_colored(self, text_widget, content, tag):
        """Insert colored text into a text widget."""
        start_pos = text_widget.index("end-1c")
        text_widget.insert("end", content)
        end_pos = text_widget.index("end-1c")
        text_widget.tag_add(tag, start_pos, end_pos)
    
    def _run_tests(self):
        """Run the tests in a background thread."""
        self.run_btn.config(state="disabled")
        self.status_label.config(text="Running tests...", foreground="orange")
        self.console_text.delete("1.0", "end")
        self.console_text.insert("1.0", "Starting tests...\n\n")
        self.notebook.select(3)  # Switch to console tab
        
        # Run in background thread
        thread = threading.Thread(target=self._run_tests_thread, daemon=True)
        thread.start()
    
    def _run_tests_thread(self):
        """Run tests in background thread."""
        import subprocess
        import sys
        
        try:
            # Run the test script
            test_script = self.tests_dir / "test_tools.py"
            
            process = subprocess.Popen(
                [sys.executable, str(test_script)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(self.tests_dir),
            )
            
            # Stream output to console
            for line in iter(process.stdout.readline, ""):
                self.root.after(0, self._append_console, line)
            
            process.wait()
            
            if process.returncode == 0:
                self.root.after(0, self._test_complete, True)
            else:
                self.root.after(0, self._test_complete, False)
                
        except Exception as e:
            self.root.after(0, self._append_console, f"\nError: {e}\n")
            self.root.after(0, self._test_complete, False)
    
    def _append_console(self, text):
        """Append text to console (called from main thread)."""
        self.console_text.insert("end", text)
        self.console_text.see("end")
    
    def _test_complete(self, success):
        """Handle test completion (called from main thread)."""
        self.run_btn.config(state="normal")
        
        if success:
            self.status_label.config(text="Tests completed!", foreground="green")
            self._load_data()  # Reload results
        else:
            self.status_label.config(text="Tests failed", foreground="red")


def main():
    root = tk.Tk()
    app = ToolTestUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
