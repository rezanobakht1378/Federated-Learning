# """yolov8-kitty-fedavg: A Flower / PyTorch app."""

# import torch
# from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
# from flwr.clientapp import ClientApp

# from yolov8_kitty_fedavg.task import Net, load_data
# from yolov8_kitty_fedavg.task import test as test_fn
# from yolov8_kitty_fedavg.task import train as train_fn

# # Flower ClientApp
# app = ClientApp()


# @app.train()
# def train(msg: Message, context: Context):
#     """Train the model on local data."""

#     # Load the model and initialize it with the received weights
#     model = Net()
#     model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     # Load the data
#     partition_id = context.node_config["partition-id"]
#     num_partitions = context.node_config["num-partitions"]
#     trainloader, _ = load_data(partition_id, num_partitions)

#     # Call the training function
#     train_loss = train_fn(
#         model,
#         trainloader,
#         context.run_config["local-epochs"],
#         msg.content["config"]["lr"],
#         device,
#     )

#     # Construct and return reply Message
#     model_record = ArrayRecord(model.state_dict())
#     metrics = {
#         "train_loss": train_loss,
#         "num-examples": len(trainloader.dataset),
#     }
#     metric_record = MetricRecord(metrics)
#     content = RecordDict({"arrays": model_record, "metrics": metric_record})
#     return Message(content=content, reply_to=msg)


# @app.evaluate()
# def evaluate(msg: Message, context: Context):
#     """Evaluate the model on local data."""

#     # Load the model and initialize it with the received weights
#     model = Net()
#     model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     # Load the data
#     partition_id = context.node_config["partition-id"]
#     num_partitions = context.node_config["num-partitions"]
#     _, valloader = load_data(partition_id, num_partitions)

#     # Call the evaluation function
#     eval_loss, eval_acc = test_fn(
#         model,
#         valloader,
#         device,
#     )

#     # Construct and return reply Message
#     metrics = {
#         "eval_loss": eval_loss,
#         "eval_acc": eval_acc,
#         "num-examples": len(valloader.dataset),
#     }
#     metric_record = MetricRecord(metrics)
#     content = RecordDict({"metrics": metric_record})
#     return Message(content=content, reply_to=msg)

"""yolov8-kitty-fedavg: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from ultralytics import YOLO
import os

# Flower ClientApp
app = ClientApp()

# -------------------------------------------------------------------
# Utility Functions for Handling YOLOv8 Model Parameters
# -------------------------------------------------------------------

def yolo_get_state_dict(model: YOLO):
    """Extract YOLOv8 PyTorch weights."""
    return model.model.state_dict()

def yolo_set_state_dict(model: YOLO, state_dict: dict):
    """Load PyTorch weights into YOLOv8."""
    model.model.load_state_dict(state_dict)

# -------------------------------------------------------------------
# TRAINING
# -------------------------------------------------------------------

@app.train()
def train(msg: Message, context: Context):
    """Train YOLOv8 on a logical dataset partition."""

    # -------------------------------------------------
    # Load YOLO model
    # -------------------------------------------------
    model = YOLO("yolov8n.pt")

    # Apply federated weights
    received_state = msg.content["arrays"].to_torch_state_dict()
    model.model.load_state_dict(received_state, strict=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.model.to(device)

    # -------------------------------------------------
    # Federated partition info
    # -------------------------------------------------
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Each client uses a deterministic fraction
    fraction = 1.0 / num_partitions
    seed = 42 + partition_id

    # -------------------------------------------------
    # Shared KITTY YAML
    # -------------------------------------------------
    yaml_path = "kitty.yaml"
    if not os.path.exists(yaml_path):
        with open(yaml_path, "w") as f:
            f.write(
                """path: .
train: images/train
val: images/val

names:
  0: car
"""
            )

    # -------------------------------------------------
    # Train YOLO (logical split, no extra folders)
    # -------------------------------------------------
    epochs = context.run_config["local-epochs"]
    lr = msg.content["config"]["lr"]

    results = model.train(
        data=yaml_path,
        epochs=epochs,
        lr0=lr,
        imgsz=640,
        batch=8,
        device=device,
        fraction=fraction,     # 🔥 key line
        seed=seed,             # 🔥 deterministic split
        verbose=False,
    )

    train_loss = float(results.results_dict["train/box_loss"])

    # -------------------------------------------------
    # Return updated weights
    # -------------------------------------------------
    model_record = ArrayRecord(model.model.state_dict())

    metrics = {
        "train_loss": train_loss,
        "num-examples": int(len(os.listdir("images/train")) * fraction),
    }

    return Message(
        content=RecordDict(
            {
                "arrays": model_record,
                "metrics": MetricRecord(metrics),
            }
        ),
        reply_to=msg,
    )

# -------------------------------------------------------------------
# EVALUATION
# -------------------------------------------------------------------

@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate YOLOv8 on the client's logical KITTY validation split."""

    # -------------------------------------------------
    # Load YOLO model
    # -------------------------------------------------
    model = YOLO("yolov8n.pt")

    # Apply federated weights
    received_state = msg.content["arrays"].to_torch_state_dict()
    model.model.load_state_dict(received_state, strict=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.model.to(device)

    # -------------------------------------------------
    # Federated partition info
    # -------------------------------------------------
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    fraction = 1.0 / num_partitions
    seed = 42 + partition_id  # same as train()

    # -------------------------------------------------
    # Shared dataset YAML
    # -------------------------------------------------
    yaml_path = "kitty.yaml"
    if not os.path.exists(yaml_path):
        with open(yaml_path, "w") as f:
            f.write(
                """path: .
train: images/train
val: images/val

names:
  0: car
"""
            )

    # -------------------------------------------------
    # Evaluate YOLO
    # -------------------------------------------------
    results = model.val(
        data=yaml_path,
        imgsz=640,
        device=device,
        fraction=fraction,   # 🔥 key line
        seed=seed,           # 🔥 deterministic client split
        verbose=False,
    )

    eval_loss = float(results.results_dict["val/box_loss"])
    eval_map = float(results.results_dict["metrics/mAP50-95(B)"])

    metrics = {
        "eval_loss": eval_loss,
        "eval_map": eval_map,
        "num-examples": int(len(os.listdir("images/val")) * fraction),
    }

    return Message(
        content=RecordDict(
            {"metrics": MetricRecord(metrics)}
        ),
        reply_to=msg,
    )