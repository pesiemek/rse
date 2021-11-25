from ser.art import generate_ascii_art
import torch

def run_inference(model, data, label):
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