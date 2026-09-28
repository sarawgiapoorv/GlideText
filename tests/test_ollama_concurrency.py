"""
tests/test_ollama_concurrency.py
Tests verifying that Ollama and FreeLLMAPI background server management
is strictly thread-safe and free from race conditions or duplicate subprocess spawns.
"""

import unittest
from unittest.mock import MagicMock, patch
import threading
import time

from local_llm import LocalLLMEngine
import freellm_manager


class TestConcurrencyAndLocks(unittest.TestCase):
    def test_ollama_concurrent_ensure_server_running(self):
        """Concurrent calls to ensure_server_running should not trigger duplicate process launches."""
        engine = LocalLLMEngine()

        spawn_count = 0
        spawn_lock = threading.Lock()

        def fake_popen(*args, **kwargs):
            nonlocal spawn_count
            with spawn_lock:
                spawn_count += 1
            time.sleep(0.05)  # Simulate slow startup
            mock_proc = MagicMock()
            mock_proc.poll.return_value = None
            return mock_proc

        # Mock is_server_running: returns False on first 2 calls, then True
        probe_count = 0
        def fake_is_running():
            nonlocal probe_count
            probe_count += 1
            return probe_count > 2

        with patch.object(engine, "_find_ollama_executable", return_value="C:\\dummy\\ollama.exe"), \
             patch.object(engine, "is_server_running", side_effect=fake_is_running), \
             patch("subprocess.Popen", side_effect=fake_popen):

            threads = []
            for _ in range(8):
                t = threading.Thread(target=engine.ensure_server_running)
                threads.append(t)

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Popen should be called at most once despite 8 simultaneous threads
            self.assertEqual(spawn_count, 1)

    def test_freellm_manager_concurrent_start(self):
        """Concurrent calls to freellm_manager.start should only spawn npm once."""
        spawn_count = 0
        spawn_lock = threading.Lock()

        def fake_popen(*args, **kwargs):
            nonlocal spawn_count
            with spawn_lock:
                spawn_count += 1
            time.sleep(0.05)
            mock_proc = MagicMock()
            mock_proc.poll.return_value = None
            return mock_proc

        probe_count = 0
        def fake_tcp_open(*args, **kwargs):
            nonlocal probe_count
            probe_count += 1
            return probe_count > 2

        with patch("freellm_manager._tcp_open", side_effect=fake_tcp_open), \
             patch("freellm_manager.locate_freellmapi_dir", return_value="C:\\dummy\\freellmapi"), \
             patch("freellm_manager._find_npm", return_value="C:\\dummy\\npm.cmd"), \
             patch("subprocess.Popen", side_effect=fake_popen):

            threads = []
            for _ in range(8):
                t = threading.Thread(target=lambda: freellm_manager.start(poll_timeout=0.2))
                threads.append(t)

            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(spawn_count, 1)


if __name__ == "__main__":
    unittest.main()
