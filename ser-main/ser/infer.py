from ser.art import generate_ascii_art
from ser.helpers import select_test_image
from ser.loaders import load_test
import torch
from torchvision import transforms

def run_inference(model, label, transform):

    image = select_test_image(label, transform)
    # run inference
    model.eval()
    output = model(image)
    pred = output.argmax(dim=1, keepdim=True)[0].item()
    certainty = torch.round(max(list(torch.exp(output)[0]))*100)


    # Draw number and show conclusion
    pixels = image[0][0]
    print(generate_ascii_art(pixels))
    print(f"I am {certainty}% sure that this is a {pred}")