"""Prevent accidental removal of mandatory manager pre-publish integrity checks."""
from pathlib import Path
import unittest

MANAGER = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "site-manager.yml"


class ManagerPublishGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = MANAGER.read_text(encoding="utf-8")
        cls.publish = cls.workflow.split("      - name: Persist repaired outputs and health status", 1)[1].split(
            "      - name: Publish repaired site directly", 1
        )[0]

    def test_stage_is_checked_before_commit(self):
        stage = self.publish.split("stage_manager_state() {", 1)[1].split("\n          }", 1)[0]
        self.assertLess(stage.index("python src/release_guard.py"), stage.index("git add outputs data/raw"))
        self.assertIn("stage_manager_state || exit 2", self.publish)

    def test_git_merge_is_checked_before_push(self):
        merge = self.publish.split("merge_latest_main() {", 1)[1].split("\n          }", 1)[0]
        self.assertIn("git merge --no-edit -X ours origin/main || return 2", merge)
        self.assertIn("python src/release_guard.py", merge)
        self.assertLess(merge.index("git merge --no-edit"), merge.rindex("python src/release_guard.py"))
        self.assertIn("merge_latest_main || exit 2", self.publish)

    def test_each_push_attempt_is_checked(self):
        attempts = self.publish.split("for attempt in 1 2 3 4; do", 1)[1]
        self.assertLess(attempts.index("python src/release_guard.py || exit 2"),
                        attempts.index("git push origin HEAD:main"))
        self.assertIn("merge_latest_main || exit 2", attempts)


if __name__ == "__main__":
    unittest.main()
