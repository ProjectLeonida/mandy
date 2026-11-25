"""Run the native Windows digest against synthetic device-fault reports."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


BOOT = Path(__file__).resolve().parents[1] / "scripts" / "windows-boot" / "boot.ps1"
POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")
FAULT = "[rt] device-fault: "


@unittest.skipUnless(os.name == "nt" and POWERSHELL, "requires PowerShell on Windows")
class FaultContextTest(unittest.TestCase):
    def context(self, lines, newline="\n"):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "boot.log").write_text("\n".join(lines), encoding="utf-8", newline=newline)
            (root / "boot.jsonl").write_text('{"kind":"trace_open"}\n', encoding="utf-8", newline="\n")
            subprocess.run(
                [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(BOOT),
                 "-DigestOnly", "-Name", "boot", "-OutDir", folder, "-SummaryLines", "0",
                 "-Python", sys.executable],
                check=True, capture_output=True, timeout=60,
            )
            digest = (root / "boot.digest.txt").read_text(encoding="utf-8-sig")
            return [line for line in digest.splitlines() if line.startswith("fault context:")]

    def test_counts_reported_and_printed_checkpoints_separately(self):
        lines = [FAULT + "submitted dispatch seq=7 queue=0xfe (copy) shader=private"] * 10
        lines += [FAULT + "VK_ERROR_DEVICE_LOST from vkQueueSubmit (private)",
                  FAULT + "VK_ERROR_DEVICE_LOST from vkQueueWaitIdle",
                  FAULT + "description: private (partial report)",
                   FAULT + "description: second report",
                  FAULT + "queue 00AAAAAA checkpoints=2",
                  FAULT + "stage=0x80 submission=1 list=0 command=2 kind=dispatch",
                  FAULT + "ExecuteCommandLists #1 queue=00AAAAAA (compute) lists=1",
                  FAULT + "last shader pipeline=0x12 shaders=private - -",
                  FAULT + "stage=0x40 marker=00000011 (unrecognized)",
                  FAULT + "queue 00BBBBBB checkpoints=0",
                  FAULT + "ExecuteCommandLists #9 queue=00BBBBBB (copy) lists=1",
                  FAULT + "queue 00CCCCCC checkpoints=3",
                  FAULT + "stage=0x20 submission=2 list=0 command=1 kind=draw",
                  FAULT + "submission metadata unavailable"]  # log ends before two stages are printed
        self.assertEqual(self.context(lines, newline="\r\n"), [
            "fault context: api=vkQueueSubmit, query=partial, queues=3, reported=5, decoded=2, unrecognized=1"
        ])

    def test_unknown_operation_and_driver_text_do_not_become_labels(self):
        self.assertEqual(self.context([
            FAULT + "VK_ERROR_DEVICE_LOST from vkQueueSubmitPrivate (vkWaitForFences)",
            FAULT + "description: private (partial report) and more",
            FAULT + "vendor[0] private stage=0x40 marker=12 (unrecognized)",
        ]), ["fault context: api=other, query=complete, queues=0, reported=0, decoded=0, unrecognized=0"])

    def test_first_recognized_api_after_an_unknown_operation(self):
        rows = self.context([FAULT + "VK_ERROR_DEVICE_LOST from private",
                             "  " + FAULT + "VK_ERROR_DEVICE_LOST from vkCreateSwapchainKHR (DXGI)"])
        self.assertEqual(rows, [
            "fault context: api=vkCreateSwapchainKHR, query=not observed, queues=0, reported=0, decoded=0, unrecognized=0"
        ])

    def test_query_failures_in_truncated_reports(self):
        for detail, outcome in [("vkGetDeviceFaultInfoEXT counts failed (-4)", "counts failed"),
                                ("vkGetDeviceFaultInfoEXT data failed (-4)", "data failed"),
                                ("fault info collection failed", "collection failed")]:
            with self.subTest(detail=detail):
                self.assertEqual(self.context([FAULT + detail]), [
                    f"fault context: api=not observed, query={outcome}, queues=0, reported=0, decoded=0, unrecognized=0"
                ])

    def test_runtime_loss_without_diagnostics(self):
        reasons = [
            ("vkWaitSemaphores in the fence dispatcher returned VK_ERROR_DEVICE_LOST", "vkWaitSemaphores"),
            ("a fence's counter read UINT64_MAX and vkDeviceWaitIdle then returned VK_ERROR_DEVICE_LOST", "vkDeviceWaitIdle"),
            ("a Vulkan call returned VK_ERROR_DEVICE_LOST", "not observed"),
        ]
        for reason, api in reasons:
            with self.subTest(reason=reason):
                rows = self.context([f"[rt] vulkan: the device was lost ({reason}); its fences now read UINT64_MAX"])
                self.assertEqual(rows, [
                    f"fault context: api={api}, query=not observed, queues=0, reported=0, decoded=0, unrecognized=0"
                ])

    def test_normal_dispatches_and_quoted_diagnostics_do_not_create_context(self):
        self.assertEqual(self.context([FAULT + "enabled (fault info on, checkpoints on, NVIDIA diagnostics config on)",
                                       FAULT + "submitted dispatch seq=7 queue=0xfe (copy) shader=private",
                                       FAULT + "queue 00AAAAAA checkpoints=999999999999999999999",
                                       "[rt] title: " + FAULT + "description: private"]), [])


if __name__ == "__main__":
    unittest.main()
