"""yolov8-kitty-fedavg: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from ultralytics import YOLO

# Flower ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # --- Load config ---
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]

    # --- Load YOLOv8 model ---
    global_model = YOLO("yolov8n.pt")

    # IMPORTANT: Flower needs PyTorch model.state_dict(), NOT YOLO object
    torch_state = global_model.model.state_dict()
    arrays = ArrayRecord(torch_state)

    # --- Initialize FedAvg ---
    strategy = FedAvg(fraction_train=fraction_train)

    # --- Run Federated Learning ---
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    # --- Save final aggregated model ---
    print("\nSaving final YOLOv8 model to disk...")

    final_state = result.arrays.to_torch_state_dict()

    # Create a fresh YOLOv8 model and load the state dict
    final_model = YOLO("yolov8n.pt")
    final_model.model.load_state_dict(final_state)

    # Save in PyTorch format
    torch.save(final_model.model.state_dict(), "final_yolov8_model.pt")

    # ALSO save in YOLO .pt format (optional)
    final_model.save("final_yolov8_flower.pt")

    print("Done! Model saved as final_yolov8_flower.pt")


# """yolov8-kitty-fedavg: A Flower / PyTorch app."""

# import torch
# from flwr.app import ArrayRecord, ConfigRecord, Context
# from flwr.serverapp import Grid, ServerApp
# from flwr.serverapp.strategy import FedAvg

# # from yolov8_kitty_fedavg.task import Net
# from ultralytics import YOLO

# # Create ServerApp
# app = ServerApp()


# @app.main()
# def main(grid: Grid, context: Context) -> None:
#     """Main entry point for the ServerApp."""

#     # Read run config
#     fraction_train: float = context.run_config["fraction-train"]
#     num_rounds: int = context.run_config["num-server-rounds"]
#     lr: float = context.run_config["lr"]

#     # Load global model
#     # global_model = Net()
#     global_model = YOLO("yolov8n.pt")
#     arrays = ArrayRecord(global_model.state_dict())

#     # Initialize FedAvg strategy
#     strategy = FedAvg(fraction_train=fraction_train)

#     # Start strategy, run FedAvg for `num_rounds`
#     result = strategy.start(
#         grid=grid,
#         initial_arrays=arrays,
#         train_config=ConfigRecord({"lr": lr}),
#         num_rounds=num_rounds,
#     )

#     # Save final model to disk
#     print("\nSaving final model to disk...")
#     state_dict = result.arrays.to_torch_state_dict()
#     torch.save(state_dict, "final_model.pt")
