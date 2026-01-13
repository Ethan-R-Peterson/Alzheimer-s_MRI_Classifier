import torch
import torchvision.transforms as T
from torchvision.models import vgg16, resnet50

CLASS_NAMES = ["Mild Impairment", "Moderate Impairment", "No Impairment", "Very Mild Impairment"]

#TODO: Transformations
def get_inference_transform():
    return T.Compose([
        T.Grayscale(num_output_channels=3),
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

MODEL_ARCHS = {
    "vgg16": vgg16,
    "resnet50": resnet50,
}

def load_model(arch: str, weights_path: str):
    if arch not in MODEL_ARCHS:
        raise ValueError(f"Unsupported architecture: {arch}")

    model_fn = MODEL_ARCHS[arch]
    model = model_fn(weights=None)

    # do 4-class classification for each model/architecture
    if arch == "vgg16":
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 4)
    elif arch == "resnet50":
        model.fc = torch.nn.Linear(model.fc.in_features, 4)

    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    if arch == "vgg16":
        def replace_relu_with_out_of_place(module):
            for name, child in module.named_children():
                if isinstance(child, torch.nn.ReLU) and child.inplace:
                    setattr(module, name, torch.nn.ReLU(inplace=False))
                else:
                    replace_relu_with_out_of_place(child)

        replace_relu_with_out_of_place(model)
        
    model.eval()
    return model

RESNET50_MODEL = load_model("resnet50", "models/best_resnet_mri.pth")
VGG16_MODEL = load_model("vgg16", "models/best_vgg_mri.pth")

##TODO: Single Model Predictions
def generate_softmax_outputs(model, input_tensor):
    with torch.no_grad():
        logits = model(input_tensor)
        softmax = torch.nn.functional.softmax(logits, dim=1)[0]

    return softmax

##TODO: Ensemble Predicitons
def plain_average_ensemble(input_tensor, models=[RESNET50_MODEL, VGG16_MODEL]):
    total_probs = torch.zeros(4)  
    with torch.no_grad():
        for model in models:
            logits = model(input_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)[0]
            total_probs += probs

    avg_probs = total_probs / len(models)
    pred_idx = int(avg_probs.argmax().item())
    pred_class = CLASS_NAMES[pred_idx]
    return pred_class, avg_probs

def weighted_average_ensemble(input_tensor, models=[RESNET50_MODEL, VGG16_MODEL], weights=[0.4, 0.6]):
    if len(models) != len(weights):
        raise ValueError("Number of models and weights must match")
    
    total_probs = torch.zeros(4)
    with torch.no_grad():
        for model, weight in zip(models, weights):
            logits = model(input_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)[0]
            total_probs += weight * probs

    pred_idx = int(total_probs.argmax().item())
    pred_class = CLASS_NAMES[pred_idx]
    return pred_class, total_probs

