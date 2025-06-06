from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import tkinter as tk
from tkinter import ttk
import requests
import traceback
import queue
from datetime import datetime
from pathlib import Path
from threading import Thread, Event
from typing import Any, Dict, List, Tuple, Optional, Union, Callable, Awaitable
from html.parser import HTMLParser

from tkinter import filedialog, messagebox, scrolledtext, simpledialog

try:
    import aiohttp
    ASYNC_MODE = True
except ImportError:
    ASYNC_MODE = False

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Runtime Globals
RUNTIME_API_KEY: Optional[str] = None

# Runtime Paths
HOME = Path.home()
ARCHIVE_DIR = HOME / "Documents" / "CatGPT_Agent_Archive"
PLUGIN_DIR = ARCHIVE_DIR / "plugins"
AGENT_WORKSPACE_DIR = ARCHIVE_DIR / "autonomous_workspace"
MEMORY_FILE = ARCHIVE_DIR / "memory.json"
MODEL_FILE = ARCHIVE_DIR / "models.json"
ARCHIVE_INDEX = ARCHIVE_DIR / "archive_index.json"

ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
PLUGIN_DIR.mkdir(parents=True, exist_ok=True)
AGENT_WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

# Constants
OPENROUTER_API_KEY_ENV_VAR = "OPENROUTER_API_KEY"
DEFAULT_MODELS = ["meta-llama/llama-3-8b-instruct", "gpt-3.5-turbo", "claude-3-opus", "gpt-4"]
LLM_TIMEOUT = 120
CODE_EXEC_TIMEOUT = 60

UI_THEME = {
    "bg_primary": "#f5f5f5", "bg_secondary": "#ffffff", "bg_tertiary": "#f0e6ff",
    "bg_chat_display": "#fafafa", "bg_chat_input": "#f9f9f9", "bg_editor": "#1e2838",
    "bg_editor_header": "#34495e", "bg_button_primary": "#10a37f", "bg_button_secondary": "#3498db",
    "bg_button_danger": "#e74c3c", "bg_button_warning": "#f39c12", "bg_button_evolution": "#9b59b6",
    "bg_button_evo_compile": "#27ae60", "bg_button_info": "#5dade2", "bg_listbox_select": "#6c5ce7",
    "bg_mission_control": "#2c3e50", "fg_mission_control": "#ecf0f1",
    "fg_primary": "#2c3e50", "fg_secondary": "#ecf0f1", "fg_button_light": "#ffffff",
    "fg_evolution_header": "#6c5ce7", "font_default": ("Consolas", 11), "font_chat": ("Consolas", 11),
    "font_button_main": ("Arial", 11, "bold"), "font_button_small": ("Arial", 10),
    "font_title": ("Arial", 14, "bold"), "font_editor": ("Consolas", 11), "font_listbox": ("Consolas", 9),
    "font_mission_control": ("Consolas", 10)
}

# Utility Helpers
def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")

def get_api_key() -> str:
    if RUNTIME_API_KEY is not None:
        return RUNTIME_API_KEY
    logger.warning("API Key not found in memory. Ensure it was set at startup.")
    return ""

def get_current_source_code() -> str:
    try:
        with open(__file__, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        try:
            return inspect.getsource(sys.modules[__name__])
        except Exception:
            return "# Error: Could not retrieve current source code."

# Custom HTML Parser for Web Scraping
class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []

    def handle_data(self, data):
        self.text.append(data)

    def get_text(self):
        return ' '.join(self.text).strip()

# Optimized API Client
class APIClient:
    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_key_getter: Callable[[], str], timeout: int):
        self._api_key_getter = api_key_getter
        self.timeout = timeout

    def _get_headers(self) -> Dict[str, str]:
        api_key = self._api_key_getter()
        if not api_key:
            raise RuntimeError("API Key is missing. Please configure it.")
        return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def _parse_response(self, data: Dict[str, Any]) -> str:
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Unexpected LLM API response structure: {data}. Error: {e}")
            raise RuntimeError("Invalid response structure from LLM API.")

    async def call_async(self, payload: Dict[str, Any]) -> str:
        async with aiohttp.ClientSession() as session:
            try:
                headers = self._get_headers()
                async with session.post(self.API_URL, headers=headers, json=payload, timeout=self.timeout) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(f"LLM API call failed with status {resp.status}: {error_text}")
                        raise RuntimeError(f"API Error (Status {resp.status}): {error_text}")
                    return self._parse_response(await resp.json())
            except aiohttp.ClientError as e:
                raise RuntimeError(f"Network Error: {e}")

    def call_sync(self, payload: Dict[str, Any]) -> str:
        try:
            response = requests.post(self.API_URL, headers=self._get_headers(), json=payload, timeout=self.timeout)
            response.raise_for_status()
            return self._parse_response(response.json())
        except requests.RequestException as e:
            raise RuntimeError(f"Network Error: {e}")

    async def close_session(self):
        pass

# Code Interpreter for Sandboxed Execution
class CodeInterpreter:
    def __init__(self, timeout: int = CODE_EXEC_TIMEOUT, workspace_dir: Path = AGENT_WORKSPACE_DIR):
        self.timeout = timeout
        self.workspace_dir = workspace_dir
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Code interpreter will use workspace: {self.workspace_dir}")

    def execute_code(self, code_string: str) -> Tuple[str, str, Optional[str], Optional[str]]:
        stdout_str, stderr_str, error_msg, saved_png_path = "", "", None, None
        with tempfile.TemporaryDirectory(dir=str(self.workspace_dir)) as temp_dir:
            temp_script_path = Path(temp_dir) / "script.py"
            try:
                temp_script_path.write_text(code_string, encoding="utf-8")
                process = subprocess.run(
                    [sys.executable, "-u", str(temp_script_path)],
                    capture_output=True, text=True, timeout=self.timeout,
                    cwd=temp_dir, check=False
                )
                stdout_str, stderr_str = process.stdout, process.stderr
                png_files = [f for f in os.listdir(temp_dir) if f.endswith('.png')]
                if png_files:
                    png_path = Path(temp_dir) / png_files[0]
                    saved_png_path = self.workspace_dir / f"output_{now_ts()}.png"
                    shutil.copy(png_path, saved_png_path)
                    saved_png_path = str(saved_png_path)
            except subprocess.TimeoutExpired:
                error_msg = f"Code execution timed out after {self.timeout} seconds."
                stderr_str += f"\nTimeoutError: Execution exceeded {self.timeout} seconds."
            except Exception as e:
                error_msg = f"An unexpected error occurred: {e}"
                logger.error(f"Code execution error: {e}")
            finally:
                if temp_script_path.exists():
                    temp_script_path.unlink()
        return stdout_str, stderr_str, error_msg, saved_png_path

# AutonomousAgent (AutoGPT Core Logic)
class AutonomousAgent:
    MASTER_TOOL_LIBRARY = {
        "execute_python_code": {
            "description": "Executes a string of Python code in a sandboxed environment. Returns stdout, stderr, and any system errors.",
            "args": {"code_string": "The Python code to execute as a single string."}
        },
        "write_file": {
            "description": "Writes content to a specified file in the agent's workspace. Overwrites if it exists.",
            "args": {"filename": "The name of the file.", "content": "The content to write."}
        },
        "read_file": {
            "description": "Reads the entire content of a specified file from the agent's workspace.",
            "args": {"filename": "The name of the file to read."}
        },
        "list_files": {
            "description": "Lists all files and directories recursively in the agent's workspace.",
            "args": {}
        },
        "search_web": {
            "description": "Fetches the content of a URL. Returns the text content of the page.",
            "args": {"url": "The full URL to retrieve."}
        },
        "task_complete": {
            "description": "Call this function ONLY when you have fully completed the user's goal. This shuts down the agent.",
            "args": {"reason": "A summary of why you believe the task is complete."}
        }
    }

    def __init__(self, goal: str, api_client: APIClient, code_interpreter: CodeInterpreter,
                 model_cfg: Dict, ui_queue: queue.Queue, stop_event: Event,
                 system_prompt: str, selected_tool_names: List[str]):
        self.goal = goal
        self.api_client = api_client
        self.code_interpreter = code_interpreter
        self.model_cfg = model_cfg
        self.ui_queue = ui_queue
        self.stop_event = stop_event
        self.system_prompt_template = system_prompt
        self.history: List[Dict[str, str]] = []

        tool_function_map = {
            "execute_python_code": self.code_interpreter.execute_code,
            "write_file": self.write_file,
            "read_file": self.read_file,
            "list_files": self.list_files,
            "search_web": self.search_web,
            "task_complete": self.task_complete
        }

        self.tools = {name: tool_function_map[name] for name in selected_tool_names if name in tool_function_map}
        self.tool_definitions = [
            {"name": name, "description": self.MASTER_TOOL_LIBRARY[name]["description"], "args": self.MASTER_TOOL_LIBRARY[name]["args"]}
            for name in selected_tool_names if name in self.MASTER_TOOL_LIBRARY
        ]

    def log_to_ui(self, message: str, tag: str = "info"):
        self.ui_queue.put({"tag": tag, "content": message})

    def get_system_prompt(self) -> str:
        base_prompt = (
            "{custom_prompt}\n\n"
            "Your primary goal is: {goal}\n\n"
            "You will operate in a continuous loop of Thought -> Plan -> Command -> Result. This loop will repeat until your goal is achieved or you are stopped. "
            "You MUST continue working, breaking down the problem and executing steps one by one until you decide the task is finished.\n\n"
            "**THE PROCESS:**\n"
            "1. **Thought:** First, deeply reflect on your main goal and the result of your last action. Reason about the most logical and effective next step to get closer to your goal. Always be critical of your own work.\n"
            "2. **Plan:** Formulate a short, numbered list of steps for your immediate next action. Be concise.\n"
            "3. **Command:** Issue a single command from your available tools. You MUST format your response as a single, valid JSON object with 'thought', 'plan', and 'command' keys. The 'command' must be an object with 'name' and 'args' keys. Do not ask for permission, just execute.\n\n"
            "**AVAILABLE TOOLS:**\n{tools}\n\n"
            "**RESPONSE FORMAT (JSON ONLY):**\n"
            "```json\n"
            "{{\n"
            '  "thought": "Your detailed reasoning for the next action. Analyze the previous result and decide what to do now.",\n'
            '  "plan": [\n'
            '    "Step 1: First part of the immediate action.",\n'
            '    "Step 2: Second part of the immediate action."\n'
            '  ],\n'
            '  "command": {{\n'
            '    "name": "tool_name",\n'
            '    "args": {{"arg1": "value1"}}\n'
            '  }}\n'
            "}}\n"
            "```\n"
            "Continuously iterate through this loop. Use the results of your actions to inform your next thought. "
            "When, and only when, the goal is fully and completely satisfied, you must call the `task_complete` function. "
            "Your first task is to think about how to achieve your goal: '{goal}'. Start by outlining a high-level plan."
        )
        return base_prompt.format(
            custom_prompt=self.system_prompt_template,
            goal=self.goal,
            tools=json.dumps(self.tool_definitions, indent=2)
        )

    def write_file(self, filename: str, content: str) -> str:
        try:
            (AGENT_WORKSPACE_DIR / filename).write_text(content, encoding='utf-8')
            return f"Successfully wrote to '{filename}'."
        except Exception as e:
            return f"Error writing to file: {e}"

    def read_file(self, filename: str) -> str:
        try:
            return (AGENT_WORKSPACE_DIR / filename).read_text(encoding='utf-8')
        except FileNotFoundError:
            return f"Error: File '{filename}' not found."
        except Exception as e:
            return f"Error reading file: {e}"

    def list_files(self) -> str:
        try:
            files = [str(p.relative_to(AGENT_WORKSPACE_DIR)) for p in AGENT_WORKSPACE_DIR.rglob("*")]
            return "Workspace files:\n" + "\n".join(files) if files else "Workspace is empty."
        except Exception as e:
            return f"Error listing files: {e}"

    def search_web(self, url: str) -> str:
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            parser = TextExtractor()
            parser.feed(response.text)
            text = parser.get_text()
            return text[:4000]
        except requests.RequestException as e:
            return f"Error searching web: {e}"

    def task_complete(self, reason: str) -> str:
        self.stop_event.set()
        return f"TASK COMPLETE. Agent shutting down. Reason: {reason}"

    def run(self):
        self.log_to_ui(f"AUTONOMOUS AGENT ACTIVATED\nGOAL: {self.goal}\nMODEL: {self.model_cfg['model']}", "header")
        self.history.append({"role": "system", "content": self.get_system_prompt()})
        self.log_to_ui(f"SYSTEM PROMPT INITIALIZED:\n{self.get_system_prompt()}", "llm")
        
        iteration_count = 0
        while not self.stop_event.is_set():
            iteration_count += 1
            self.log_to_ui(f"--- Starting Iteration {iteration_count} ---", "status")
            
            try:
                self.log_to_ui("Thinking...", "status")
                payload = self.model_cfg.copy()
                payload['messages'] = [self.history[0]] + self.history[-10:]
                
                llm_response_raw = self.api_client.call_sync(payload)
                self.log_to_ui(f"LLM Raw Response:\n{llm_response_raw}", "llm")
                
                match = re.search(r"```json\s*\n(.*?)\n```", llm_response_raw, re.DOTALL)
                if not match:
                    error_feedback = "Your response was not in the required JSON format. Please enclose the JSON object in ```json ... ``` tags and try again."
                    self.log_to_ui(f"ERROR: LLM did not return a valid JSON code block. Instructing to retry.", "error")
                    self.history.append({"role": "assistant", "content": llm_response_raw})
                    self.history.append({"role": "user", "content": error_feedback})
                    continue

                parsed_json = json.loads(match.group(1).strip())
                thought = parsed_json.get("thought", "No thought provided.")
                plan = "\n".join(f"- {p}" for p in parsed_json.get("plan", []))
                command = parsed_json.get("command", {})
                command_name = command.get("name")
                command_args = command.get("args", {})

                self.log_to_ui(f"THOUGHT: {thought}\nPLAN:\n{plan}\nCOMMAND: {command_name}({json.dumps(command_args)})", "thought")
                self.history.append({"role": "assistant", "content": llm_response_raw})

                user_feedback = ""
                if command_name in self.tools:
                    tool_func = self.tools[command_name]
                    if command_name == 'execute_python_code':
                        stdout, stderr, exec_err, png_path = tool_func(**command_args)
                        result = f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
                        if exec_err:
                            result += f"\nEXECUTION_ERROR: {exec_err}"
                        if png_path:
                            result += f"\nPNG generated: {png_path}"
                    else:
                        result = tool_func(**command_args)
                    
                    self.log_to_ui(f"RESULT:\n{result}", "result")
                    user_feedback = f"Command `{command_name}` executed. Result was:\n{result}"
                else:
                    result = f"Error: Unknown command '{command_name}'."
                    self.log_to_ui(result, "error")
                    user_feedback = f"The command '{command_name}' does not exist. Review your plan and choose a valid command from this list: {list(self.tools.keys())}"
                
                self.history.append({"role": "user", "content": user_feedback})
                self.log_to_ui("Looping for next action...", "status")

            except json.JSONDecodeError as e:
                self.log_to_ui(f"ERROR: Failed to decode JSON from LLM response: {e}", "error")
                error_feedback = "Your last response was not valid JSON. Please correct the JSON structure and try again."
                self.history.append({"role": "user", "content": error_feedback})
            except Exception as e:
                tb = traceback.format_exc()
                self.log_to_ui(f"CRITICAL AGENT ERROR: {e}\n{tb}", "error")
                self.stop_event.set()

        self.log_to_ui("Autonomous agent has shut down.", "header")

# DarwinAgent (Interactive Chat Agent)
class DarwinAgent:
    def __init__(self, ui_app_ref):
        self.ui_app = ui_app_ref
        self.models: List[str] = self._load_models()
        self.cfg: Dict[str, Any] = {"model": self.models[0], "temperature": 0.7, "max_tokens": 4096}
        self.history: List[Dict[str, str]] = self._load_memory()
        self.agent_archive: List[Tuple] = self._load_agent_archive()
        logger.info(f"Loaded {len(self.agent_archive)} items from agent archive.")
        self.api_client = APIClient(get_api_key, LLM_TIMEOUT)
        self.code_interpreter = CodeInterpreter(timeout=CODE_EXEC_TIMEOUT)
        logger.info("DarwinAgent initialized.")

    def _load_json_file(self, fp, default):
        return json.loads(fp.read_text()) if fp.exists() else default

    def _save_json_file(self, fp, data):
        fp.write_text(json.dumps(data, indent=2))

    def _load_memory(self):
        return self._load_json_file(MEMORY_FILE, [])

    def _save_memory(self):
        self._save_json_file(MEMORY_FILE, self.history[-2000:])

    def _load_models(self):
        models = self._load_json_file(MODEL_FILE, [])
        if not models:
            self._save_json_file(MODEL_FILE, DEFAULT_MODELS)
            return DEFAULT_MODELS
        return models

    def _load_agent_archive(self):
        return self._load_json_file(ARCHIVE_INDEX, [])

    def _save_agent_archive(self):
        self._save_json_file(ARCHIVE_INDEX, self.agent_archive)
        logger.info(f"Saved agent archive with {len(self.agent_archive)} items.")

    def _prepare_payload(self, custom_messages: Optional[List[Dict]] = None) -> Dict[str, Any]:
        messages_to_send = custom_messages if custom_messages is not None else self.history
        system_prompt = ("You are DarwinCat, a helpful AI assistant. To execute Python code, respond with ```python_exec...```.")
        final_messages = list(messages_to_send)
        if not final_messages or final_messages[0].get("role") != "system":
            final_messages.insert(0, {"role": "system", "content": system_prompt})
        return {"model": self.cfg["model"], "messages": final_messages[-20:], "temperature": self.cfg["temperature"], "max_tokens": self.cfg["max_tokens"]}

    def _extract_executable_code(self, msg: str) -> Optional[str]:
        return (m.group(1).strip() if (m := re.search(r"```python_exec\s*\n(.*?)\n```", msg, re.DOTALL)) else None)

    def _format_code_execution_summary(self, stdout: str, stderr: str, exec_error: Optional[str], png_path: Optional[str]) -> str:
        summary = "--- Python Code Execution ---\n"
        if stdout:
            summary += f"STDOUT:\n```text\n{stdout.strip()}\n```\n"
        if stderr:
            summary += f"STDERR:\n```text\n{stderr.strip()}\n```\n"
        if exec_error:
            summary += f"EXECUTION SYSTEM ERROR: {exec_error}\n"
        if png_path:
            summary += f"PNG generated: {png_path}\n"
        if not (stdout or stderr or exec_error or png_path):
            summary += "Code executed successfully with no output.\n"
        return summary.strip()

    async def generate_agent_design(self, goal: str) -> str:
        tool_defs = [
            {"name": name, "description": details["description"]}
            for name, details in AutonomousAgent.MASTER_TOOL_LIBRARY.items()
        ]
        metaprompt = (
            "You are an expert AI agent architect. Your task is to design a specialized autonomous agent based on a user's goal. "
            "You must define a custom system prompt for the agent and select the most appropriate tools from the provided list.\n\n"
            "USER'S GOAL: '{goal}'\n\n"
            "AVAILABLE TOOLS:\n{tools}\n\n"
            "INSTRUCTIONS:\n"
            "1.  **system_prompt:** Write a concise, motivating system prompt for the new agent. Give it a persona and clear instructions relevant to the goal. DO NOT just repeat the user's goal.\n"
            "2.  **tools:** From the list of available tools, select the names of the tools the agent will absolutely need to achieve the goal. Be economical; do not select tools that are not necessary.\n\n"
            "Respond with a single, valid JSON object containing the keys 'system_prompt' and 'tools'.\n"
            "Example Response:\n"
            "```json\n"
            "{{\n"
            '  "system_prompt": "You are a web researcher agent. Your mission is to find information and synthesize it into a report.",\n'
            '  "tools": ["search_web", "write_file", "task_complete"]\n'
            "}}\n"
            "```"
        ).format(goal=goal, tools=json.dumps(tool_defs, indent=2))
        payload = self._prepare_payload(custom_messages=[{"role": "user", "content": metaprompt}])
        payload['temperature'] = 0.2
        response = await self.api_client.call_async(payload)
        match = re.search(r"```json\s*\n(.*?)\n```", response, re.DOTALL)
        if not match:
            raise RuntimeError("Agent architect failed to return valid JSON.")
        return match.group(1).strip()

    async def _ask_orchestrator(self, user_msg: str) -> str:
        if user_msg.startswith("/model "):
            mdl = user_msg.split(maxsplit=1)[1]
            self.cfg["model"] = mdl
            return f"Model switched to {mdl}"
        if user_msg.startswith("/imagine "):
            goal = user_msg.split(maxsplit=1)[1]
            try:
                self.ui_app.after(0, self.ui_app.set_status, f"Designing agent for goal: {goal}...")
                agent_design_json = await self.generate_agent_design(goal)
                agent_design = json.loads(agent_design_json)
                system_prompt = agent_design['system_prompt']
                selected_tools = agent_design['tools']
                self.ui_app.after(0, self.ui_app.launch_autonomous_agent_window, goal, system_prompt, selected_tools)
                return (f"**AUTONOMOUS AGENT DESIGNED & DISPATCHED!**\n\n"
                        f"**Goal:** '{goal}'\n"
                        f"**Agent Persona:** {system_prompt}\n"
                        f"**Tools:** {', '.join(selected_tools)}\n\n"
                        "*A new 'Mission Control' window has opened to monitor the agent.*")
            except Exception as e:
                logger.error(f"Error during agent generation: {e}\n{traceback.format_exc()}")
                return f"[Agent-Generation-Error] {e}"
        
        self.history.append({"role": "user", "content": user_msg})
        try:
            payload = self._prepare_payload()
            assistant_msg_1 = await self.api_client.call_async(payload)
            self.history.append({"role": "assistant", "content": assistant_msg_1})
            if code_to_execute := self._extract_executable_code(assistant_msg_1):
                stdout, stderr, exec_err, png_path = await asyncio.to_thread(self.code_interpreter.execute_code, code_to_execute)
                summary = self._format_code_execution_summary(stdout, stderr, exec_err, png_path)
                feedback = f"The Python code was executed. Result:\n{summary}"
                self.history.append({"role": "user", "content": feedback})
                payload_2 = self._prepare_payload()
                assistant_msg_2 = await self.api_client.call_async(payload_2)
                self.history.append({"role": "assistant", "content": assistant_msg_2})
                self._save_memory()
                return f"{assistant_msg_1}\n\n{summary}\n\n**CatGPT (after execution):**\n{assistant_msg_2}"
            self._save_memory()
            return assistant_msg_1
        except RuntimeError as e:
            return f"[LLM-Error] {e}"
        except Exception as e:
            logger.error(f"Unexpected error in orchestrator: {e}\n{traceback.format_exc()}")
            return f"[Agent-Error] An unexpected error occurred: {e}"

    async def ask_async(self, user_msg: str) -> str:
        return await self._ask_orchestrator(user_msg)

    def ask_sync(self, user_msg: str) -> str:
        return asyncio.run(self._ask_orchestrator(user_msg))

    def recompile(self, new_code: str) -> Optional[Tuple[str, str, Optional[str]]]:
        ts = now_ts()
        initial_filename = f"CatGPT_Agent_v{ts}.py"
        save_path_str = filedialog.asksaveasfilename(
            initialdir=ARCHIVE_DIR,
            initialfile=initial_filename,
            defaultextension=".py",
            filetypes=[("Python Files", "*.py"), ("All Files", "*.*")],
            title="Save Evolved Agent As..."
        )
        if not save_path_str:
            return None
        agent_file_path = Path(save_path_str)
        status, error = ("FIT", None) if "class DarwinAgent" in new_code else ("QUARANTINED", "Missing DarwinAgent class")
        final_code = f'"""\nTimestamp: {datetime.now()}\nStatus: {status}\n"""\n\n' + new_code
        try:
            agent_file_path.write_text(final_code, encoding="utf-8")
            self.agent_archive.append((ts, agent_file_path.name, status, str(error)))
            self._save_agent_archive()
            return agent_file_path.name, status, str(error)
        except Exception as e:
            logger.error(f"Failed to save recompiled agent: {e}")
            messagebox.showerror("Save Error", f"Could not save the agent file.\nError: {e}")
            return None

    async def shutdown(self):
        await self.api_client.close_session()

# Tkinter UI Layer
class CatGPTFusion(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Darwin CatGPT Fusion Edition")
        self.geometry("1100x750")
        self.config(bg=UI_THEME["bg_primary"])
        self.shutdown_event = Event()
        self.threads = []
        self._prompt_for_api_key_if_missing()
        self.agent = DarwinAgent(self)
        self.intro_message = "Welcome to Darwin CatGPT! Use /imagine <your goal> to design and launch an autonomous agent."
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        logger.info("CatGPTFusion UI initialized.")

    def _prompt_for_api_key_if_missing(self):
        global RUNTIME_API_KEY
        if env_key := os.environ.get(OPENROUTER_API_KEY_ENV_VAR):
            RUNTIME_API_KEY = env_key.strip()
        else:
            RUNTIME_API_KEY = (simpledialog.askstring("API Key Required", "Enter your OpenRouter API Key:") or "").strip()
        if not RUNTIME_API_KEY:
            messagebox.showwarning("API Key Missing", "No API Key entered. AI features may fail.")

    def _build_ui(self):
        main = tk.Frame(self, bg=UI_THEME["bg_primary"])
        main.pack(fill="both", expand=True, padx=10, pady=10)
        notebook = ttk.Notebook(main)
        notebook.pack(fill="both", expand=True)

        # Chat Tab
        chat_tab = tk.Frame(notebook, bg=UI_THEME["bg_secondary"])
        notebook.add(chat_tab, text="Chat")
        left = tk.Frame(chat_tab, bg=UI_THEME["bg_secondary"], relief=tk.RAISED, bd=1)
        left.pack(side="left", fill="both", expand=True, padx=(0, 5))
        right = tk.Frame(chat_tab, bg=UI_THEME["bg_tertiary"], relief=tk.RIDGE, bd=2)
        right.pack(side="right", fill="y", padx=(5, 0))
        self._build_chat_display(left)
        self._build_input_area(left)
        self._build_control_buttons(left)
        self._build_archive_panel(right)

        # Code Editor Tab
        code_tab = tk.Frame(notebook, bg=UI_THEME["bg_editor"])
        notebook.add(code_tab, text="Code Editor")
        self.code_editor = tk.Text(code_tab, bg=UI_THEME["bg_editor"], fg=UI_THEME["fg_secondary"], font=UI_THEME["font_editor"])
        self.code_editor.pack(fill="both", expand=True, padx=5, pady=5)
        self.code_output = scrolledtext.ScrolledText(code_tab, bg=UI_THEME["bg_chat_display"], font=UI_THEME["font_chat"])
        self.code_output.pack(fill="both", expand=True, padx=5, pady=5)
        tk.Button(code_tab, text="Run Code", command=self.run_code, bg=UI_THEME["bg_button_primary"], fg=UI_THEME["fg_button_light"]).pack(pady=5)

        # Archive History Tab
        archive_history_tab = tk.Frame(notebook, bg=UI_THEME["bg_secondary"])
        notebook.add(archive_history_tab, text="Archive History")
        self._build_archive_history(archive_history_tab)

        self.status_bar = tk.Label(self, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W, bg=UI_THEME["bg_primary"], fg=UI_THEME["fg_primary"])
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        self._display_initial_messages()

    def _build_chat_display(self, p):
        self.chat_window = scrolledtext.ScrolledText(p, bg=UI_THEME["bg_chat_display"], font=UI_THEME["font_chat"], wrap=tk.WORD, relief=tk.SOLID, bd=1)
        self.chat_window.pack(fill="both", expand=True, padx=10, pady=10)
        self.chat_window.configure(state=tk.DISABLED)
        self.chat_window.tag_config("user_tag", font=(*UI_THEME["font_chat"], "bold"), foreground="#2980b9")
        self.chat_window.tag_config("gpt_tag", font=(*UI_THEME["font_chat"], "bold"), foreground="#8e44ad")
        self.chat_window.tag_config("bold", font=(*UI_THEME["font_chat"], "bold"))
        self.chat_window.tag_config("italic", font=(*UI_THEME["font_chat"], "italic"))

    def _build_input_area(self, p):
        f = tk.Frame(p, bg=UI_THEME["bg_secondary"])
        f.pack(fill="x", padx=10, pady=(0, 10))
        self.input_field = tk.Text(f, height=3, bg=UI_THEME["bg_chat_input"], font=UI_THEME["font_chat"], wrap=tk.WORD, relief=tk.SOLID, bd=1)
        self.input_field.pack(side="left", fill="x", expand=True)
        self.input_field.bind("<Return>", lambda e: self._on_send() if not (e.state & 0x1) else None)
        tk.Button(f, text="Send", command=self._on_send, bg=UI_THEME["bg_button_primary"], fg=UI_THEME["fg_button_light"], font=UI_THEME["font_button_main"]).pack(side="right", padx=5, fill="y")

    def _build_control_buttons(self, p):
        f = tk.Frame(p, bg=UI_THEME["bg_secondary"])
        f.pack(fill="x", padx=10, pady=(0, 10))
        btns = [
            ("Recompile", self._agent_recompile_window),
            ("Clear History", self._clear_history),
            ("Clear Archive", self._clear_archive)
        ]
        for txt, cmd in btns:
            tk.Button(f, text=txt, command=cmd, bg=UI_THEME["bg_button_secondary"], fg=UI_THEME["fg_button_light"]).pack(side="left", padx=2)

    def _build_archive_panel(self, p):
        tk.Label(p, text="🧬 Evolution Archive", font=UI_THEME["font_title"], bg=UI_THEME["bg_tertiary"], fg=UI_THEME["fg_evolution_header"]).pack(pady=8)
        self.archive_listbox = tk.Listbox(p, width=50, font=UI_THEME["font_listbox"], relief=tk.SOLID, bd=1)
        self.archive_listbox.pack(fill="both", expand=True, padx=5)
        self._refresh_archive_listbox()

    def _build_archive_history(self, p):
        tk.Label(p, text="Evolution Archive History", font=UI_THEME["font_title"], bg=UI_THEME["bg_secondary"], fg=UI_THEME["fg_primary"]).pack(pady=8)
        self.archive_history_listbox = tk.Listbox(p, width=100, font=UI_THEME["font_listbox"], relief=tk.SOLID, bd=1)
        self.archive_history_listbox.pack(fill="both", expand=True, padx=5, pady=5)
        self._refresh_archive_history()

    def set_status(self, text: str):
        self.status_bar.config(text=text)

    def _display_initial_messages(self):
        self.chat_window.config(state=tk.NORMAL)
        self.chat_window.insert(tk.END, "CatGPT: ", "gpt_tag")
        self.chat_window.insert(tk.END, f"{self.intro_message}\n")
        self.chat_window.insert(tk.END, "Models: ", "bold")
        self.chat_window.insert(tk.END, f"{', '.join(self.agent.models)}\n")
        self.chat_window.insert(tk.END, "Commands: ", "bold")
        self.chat_window.insert(tk.END, "/model <name>, /imagine <goal>\n")
        self.chat_window.config(state=tk.DISABLED)

    def _append_chat(self, who: str, txt: str):
        self.chat_window.config(state=tk.NORMAL)
        self.chat_window.insert(tk.END, f"\n{who}:\n", "user_tag" if who == "You" else "gpt_tag")
        for line in txt.split('\n'):
            line = line.replace('**', '')
            parts = re.split(r'(\*.*?\*)', line)
            for part in parts:
                if part.startswith('*') and part.endswith('*'):
                    self.chat_window.insert(tk.END, part[1:-1], "italic")
                else:
                    self.chat_window.insert(tk.END, part)
            self.chat_window.insert(tk.END, "\n")
        self.chat_window.see(tk.END)
        self.chat_window.config(state=tk.DISABLED)

    def _on_send(self):
        user_msg = self.input_field.get("1.0", "end-1c").strip()
        if not user_msg:
            return
        self.input_field.delete("1.0", tk.END)
        self._append_chat("You", user_msg)
        self.set_status("Thinking...")
        Thread(target=self._worker, args=(user_msg,), daemon=True).start()

    def _worker(self, msg: str):
        try:
            answer = asyncio.run(self.agent.ask_async(msg)) if ASYNC_MODE else self.agent.ask_sync(msg)
        except Exception as e:
            logger.error(f"Error in worker thread: {e}\n{traceback.format_exc()}")
            answer = f"[error] An unexpected error occurred: {e}"
        if self.winfo_exists():
            self.after(0, self.set_status, "Ready")
            self.after(0, lambda: self._append_chat("CatGPT", answer))

    def _clear_history(self):
        if messagebox.askyesno("Confirm", "Clear chat history and memory?"):
            self.agent.history.clear()
            self.agent._save_memory()
            self.chat_window.config(state=tk.NORMAL)
            self.chat_window.delete('1.0', tk.END)
            self._display_initial_messages()

    def _clear_archive(self):
        if messagebox.askyesno("Confirm", "Are you sure you want to clear the evolution archive? This will delete all archived agent files."):
            self.agent.agent_archive.clear()
            self.agent._save_agent_archive()
            for file in ARCHIVE_DIR.glob("CatGPT_Agent_v*.py"):
                try:
                    file.unlink()
                except Exception as e:
                    logger.error(f"Failed to delete file {file}: {e}")
            self._refresh_archive_listbox()
            self._refresh_archive_history()
            messagebox.showinfo("Archive Cleared", "The evolution archive has been cleared.")

    def run_code(self):
        if self.shutdown_event.is_set():
            return
        code = self.code_editor.get("1.0", tk.END).strip()
        if not code:
            return
        self.code_output.delete("1.0", tk.END)
        self.code_output.insert(tk.END, "Running code...\n")
        def execute():
            stdout, stderr, exec_err, png_path = self.agent.code_interpreter.execute_code(code)
            if not self.shutdown_event.is_set():
                self.after(0, lambda: self.display_code_output(stdout, stderr, exec_err, png_path))
        thread = Thread(target=execute, daemon=True)
        thread.start()
        self.threads.append(thread)

    def display_code_output(self, stdout: str, stderr: str, exec_err: Optional[str], png_path: Optional[str]):
        self.code_output.delete("1.0", tk.END)
        if stdout:
            self.code_output.insert(tk.END, "Output:\n" + stdout + "\n")
        if stderr:
            self.code_output.insert(tk.END, "\nErrors:\n" + stderr + "\n")
        if exec_err:
            self.code_output.insert(tk.END, f"Execution Error:\n{exec_err}\n")
        if not (stdout or stderr or exec_err):
            self.code_output.insert(tk.END, "Code executed successfully (no output).\n")
        if png_path and os.path.exists(png_path):
            self.code_output.insert(tk.END, f"\nPNG generated and saved to: {png_path}\n")
            messagebox.showinfo("PNG Generated", f"PNG saved to {png_path}")
        else:
            self.code_output.insert(tk.END, "\nNo PNG generated or file not found.\n")

    def _on_closing(self):
        logger.info("Initiating application shutdown")
        self.shutdown_event.set()
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
        self.destroy()

    def _agent_recompile_window(self):
        win = tk.Toplevel(self)
        win.title("Agent Recompile")
        win.geometry("900x700")
        win.transient(self)
        win.grab_set()
        editor = tk.Text(win, bg=UI_THEME["bg_editor"], fg=UI_THEME["fg_secondary"], font=UI_THEME["font_editor"], undo=True, insertbackground="white")
        editor.pack(fill="both", expand=True, padx=10, pady=10)
        editor.insert(tk.END, get_current_source_code())
        
        def compile_and_evolve():
            result = self.agent.recompile(editor.get("1.0", "end-1c"))
            if result:
                filename, status, _ = result
                self._refresh_archive_listbox()
                self._refresh_archive_history()
                (messagebox.showinfo if status == "FIT" else messagebox.showwarning)(
                    "Evolution Result", 
                    f"Status: {status}\nSaved as: {filename}", 
                    parent=win
                )
                win.destroy()
        
        tk.Button(win, text="🚀 Compile & Evolve", command=compile_and_evolve, bg=UI_THEME["bg_button_evo_compile"], fg=UI_THEME["fg_button_light"]).pack(pady=10)

    def _refresh_archive_listbox(self):
        self.archive_listbox.delete(0, tk.END)
        for ts, filename, status, _ in reversed(self.agent.agent_archive):
            icon = "✅" if status == "FIT" else "🔒"
            self.archive_listbox.insert(tk.END, f"{icon} {ts.split('_')[0]} | {filename}")

    def _refresh_archive_history(self):
        self.archive_history_listbox.delete(0, tk.END)
        for ts, filename, status, error in reversed(self.agent.agent_archive):
            icon = "✅" if status == "FIT" else "🔒"
            entry = f"{icon} {ts} | {filename} | Status: {status}"
            if error:
                entry += f" | Error: {error}"
            self.archive_history_listbox.insert(tk.END, entry)

    def launch_autonomous_agent_window(self, goal: str, system_prompt: str, selected_tool_names: List[str]):
        win = tk.Toplevel(self)
        win.title("CatGPT Mission Control")
        win.geometry("800x600")
        win.config(bg=UI_THEME["bg_mission_control"])
        log_display = scrolledtext.ScrolledText(win, wrap=tk.WORD, bg="#1C1C1C", fg=UI_THEME["fg_mission_control"], font=UI_THEME["font_mission_control"], insertbackground="white")
        log_display.pack(fill="both", expand=True, padx=5, pady=5)
        log_display.tag_config("header", foreground="#9b59b6", font=(*UI_THEME["font_mission_control"], "bold"))
        log_display.tag_config("status", foreground="#f39c12")
        log_display.tag_config("thought", foreground="#3498db", lmargin1=10, lmargin2=10)
        log_display.tag_config("result", foreground="#2ecc71", lmargin1=10, lmargin2=10)
        log_display.tag_config("llm", foreground="#7f8c8d", font=(*UI_THEME["font_mission_control"], "italic"), lmargin1=10, lmargin2=10)
        log_display.tag_config("error", foreground="#e74c3c", font=(*UI_THEME["font_mission_control"], "bold"))
        
        ui_queue = queue.Queue()
        stop_event = Event()

        def stop_agent(): 
            if not stop_event.is_set():
                stop_event.set()
                if stop_button.winfo_exists():
                    stop_button.config(state=tk.DISABLED, text="STOPPING...")
        
        stop_button = tk.Button(win, text="🛑 STOP AGENT", command=stop_agent, bg=UI_THEME["bg_button_danger"], fg="white", font=UI_THEME["font_button_main"])
        stop_button.pack(pady=5)
        
        agent_instance = AutonomousAgent(
            goal=goal,
            api_client=self.agent.api_client,
            code_interpreter=self.agent.code_interpreter,
            model_cfg=self.agent.cfg,
            ui_queue=ui_queue,
            stop_event=stop_event,
            system_prompt=system_prompt,
            selected_tool_names=selected_tool_names
        )
        Thread(target=agent_instance.run, daemon=True).start()

        def process_queue():
            try:
                while not ui_queue.empty():
                    msg = ui_queue.get_nowait()
                    if win.winfo_exists():
                        log_display.config(state=tk.NORMAL)
                        log_display.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] ", ("info",))
                        log_display.insert(tk.END, f"{msg['content']}\n\n", (msg['tag'],))
                        log_display.config(state=tk.DISABLED)
                        log_display.see(tk.END)
            except queue.Empty:
                pass
            if not stop_event.is_set() and win.winfo_exists():
                win.after(100, process_queue)
            elif win.winfo_exists():
                log_display.config(state=tk.NORMAL)
                log_display.insert(tk.END, "[Mission Control] Agent has stopped.\n", ("header",))
                log_display.config(state=tk.DISABLED)

        win.after(100, process_queue)

if __name__ == "__main__":
    app = CatGPTFusion()
    app.mainloop()
