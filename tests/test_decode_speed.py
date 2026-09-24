# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tools import decode_speed, soak_runtime

METRICS = """# HELP vllm:spec_decode_num_drafts_total Number of drafts.
vllm:spec_decode_num_drafts_total{engine="0",model_name="m"} 100.0
vllm:spec_decode_num_drafts_total{engine="1",model_name="m"} 20.0
vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="m"} 150.0
vllm:spec_decode_num_accepted_tokens_total_created{engine="0"} 5.0
"""


def test_spec_counters_sum_every_label_set_and_ignore_other_series():
    assert decode_speed.spec_counters(METRICS) == {
        decode_speed.DRAFTS: 120.0, decode_speed.ACCEPTED: 150.0}


def test_ms_per_step_divides_decode_time_by_draft_steps():
    run = decode_speed.decode_numbers(seconds=5.0, completion_tokens=400, drafts=125, accepted=275)

    assert run["ms_per_step"] == 40.0
    assert run["tokens_per_step"] == 3.2


def test_without_spec_metrics_a_step_is_one_token():
    run = decode_speed.decode_numbers(seconds=3.99, completion_tokens=400, drafts=0, accepted=0)

    assert run["steps"] == 399
    assert run["ms_per_step"] == 10.0


def test_first_round_is_after_start_and_the_rest_are_warm():
    rounds = [[{"ms_per_step": 47.0, "tokens_per_step": 3.0}],
              [{"ms_per_step": 34.0, "tokens_per_step": 3.0}],
              [{"ms_per_step": 36.0, "tokens_per_step": 3.0}]]

    result = decode_speed.summary(rounds)

    assert result["after_start_ms_per_step"] == [47.0]
    assert result["warm_ms_per_step"] == [34.0, 36.0]
    assert result["warm_median_ms_per_step"] == 35.0


class FakeServer(BaseHTTPRequestHandler):
    """/metrics counts 100 drafts and 200 accepted tokens per finished request;
    chat streams 4 chunks and reports 400 completion tokens."""

    requests = 0

    def log_message(self, *args):
        pass

    def do_GET(self):
        done = FakeServer.requests
        body = (f"vllm:spec_decode_num_drafts_total {100.0 * done}\n"
                f"vllm:spec_decode_num_accepted_tokens_total {200.0 * done}\n").encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert request["temperature"] == 0 and request["ignore_eos"] is True
        self.send_response(200)
        self.end_headers()
        for _ in range(4):
            self.wfile.write(b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n')
        self.wfile.write(b'data: {"choices":[],"usage":{"completion_tokens":400}}\n\ndata: [DONE]\n\n')
        FakeServer.requests += 1


@pytest.fixture
def server():
    FakeServer.requests = 0
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeServer)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()


def test_measurement_saves_after_start_and_warm_rounds(server, tmp_path):
    output = tmp_path / "decode.json"

    decode_speed.measure(server, "m", output, "profile-id", "on")

    saved = json.loads(output.read_text())
    assert saved["schema"] == "r9v.decode-speed.v1"
    assert saved["profile"] == "profile-id" and saved["ced"] == "on"
    assert len(saved["after_start_ms_per_step"]) == 3
    assert len(saved["warm_ms_per_step"]) == 6
    assert {run["tokens_per_step"] for run in saved["runs"]} == {3.0}


def test_measurement_never_overwrites_an_existing_file(server, tmp_path):
    output = tmp_path / "decode.json"
    output.write_text("keep")

    with pytest.raises(FileExistsError):
        decode_speed.measure(server, "m", output, None, None)
    assert output.read_text() == "keep"


def test_soak_decode_speed_option_reports_a_dead_server_and_exits_1(tmp_path, capsys):
    code = soak_runtime.main(["--port", "9", "--decode-speed", str(tmp_path / "d.json")])

    assert code == 1
    assert "Decode measurement failed" in capsys.readouterr().out
