from freetoken.server.args import parse_args


def _parse(*extra: str):
    # Keep this parser test independent of a checkpoint or Hub access. Auto dtype and
    # parser selection intentionally inspect config.json; none of that is relevant to
    # the rendezvous endpoint contract under test.
    return parse_args(
        [
            "--model",
            "/models/anonymous",
            "--dtype",
            "bfloat16",
            "--tool-call-parser",
            "llama3",
            "--reasoning-parser",
            "off",
            *extra,
        ]
    )


def test_distributed_port_defaults_next_to_api_port():
    args, _ = _parse("--port", "8081")
    assert args.distributed_addr == "tcp://127.0.0.1:8082"


def test_distributed_port_can_be_selected_independently():
    args, _ = _parse(
        "--port",
        "8081",
        "--distributed-port",
        "18082",
    )
    assert args.distributed_addr == "tcp://127.0.0.1:18082"
