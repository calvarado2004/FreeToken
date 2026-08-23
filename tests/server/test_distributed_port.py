from freetoken.server.args import parse_args


def test_distributed_port_defaults_next_to_api_port():
    args, _ = parse_args(["--model", "/models/anonymous", "--port", "8081"])
    assert args.distributed_addr == "tcp://127.0.0.1:8082"


def test_distributed_port_can_be_selected_independently():
    args, _ = parse_args(
        [
            "--model",
            "/models/anonymous",
            "--port",
            "8081",
            "--distributed-port",
            "18082",
        ]
    )
    assert args.distributed_addr == "tcp://127.0.0.1:18082"
