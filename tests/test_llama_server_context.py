import unittest
from unittest.mock import patch

from grading_agent.llm import server


class FakeProcess:
    returncode = None

    def poll(self):
        return None


class FakeResponse:
    status_code = 200


class LlamaServerContextTests(unittest.TestCase):
    @patch("grading_agent.llm.server.time.sleep")
    @patch("grading_agent.llm.server.requests.get", return_value=FakeResponse())
    @patch("grading_agent.llm.server.subprocess.Popen")
    @patch("grading_agent.llm.server.shutil.which", return_value="/usr/bin/llama-server")
    def test_server_starts_with_configured_context_size(self, which, popen, get, sleep):
        process = FakeProcess()
        popen.return_value = process
        with patch.object(server, "LLAMA_CONTEXT_SIZE", 8192):
            result = server.initiate_server()
        self.assertIs(result, process)
        command = popen.call_args.args[0]
        self.assertIn("--ctx-size", command)
        self.assertEqual(command[command.index("--ctx-size") + 1], "8192")


if __name__ == "__main__":
    unittest.main()
