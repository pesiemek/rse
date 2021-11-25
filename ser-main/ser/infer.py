from ser.art import generate_ascii_art
import torch
from ser.transforms import transforms, normalize, flip


def run_inference(model, data, label):
    def _select_test_image(data, gitlabel):
     dataloader = load_test(1, transforms(normalize))
     # TODO we should be able to switch between these abstractions without
     #   having to change any code.
     #   make it happen!
     ts = [normalize, flip]
     dataloader = load_test(1, transforms(*ts))
    
    images, labels = next(iter(data))
    while labels[0].item() != label:
        images, labels = next(iter(data))

    # run inference
    model.eval()
    output = model(images)
    pred = output.argmax(dim=1, keepdim=True)[0].item()
    certainty = torch.round(max(list(torch.exp(output)[0]))*100)


    # Draw number and show conclusion
    pixels = images[0][0]
    print(generate_ascii_art(pixels))
    print(f"I am {certainty}% sure that this is a {pred}")