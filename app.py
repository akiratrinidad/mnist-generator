import streamlit as st
import torch
import numpy as np
from PIL import Image
from model import VAE
from torchvision import datasets, transforms

# Load MNIST dataset (for digit reference)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

# Load the trained model
@st.cache_resource
def load_model():
    model = VAE(latent_dim=20)
    model.load_state_dict(torch.load('mnist_vae.pth', map_location='cpu'))
    model.eval()
    return model

model = load_model()

# Function to generate digit images
def generate_samples(model, digit, num_samples=5):
    # Collect real samples of the chosen digit
    digit_data = []
    for data, target in train_dataset:
        if target == digit:
            digit_data.append(data)
            if len(digit_data) >= 1000:  # Use 1000 samples to estimate distribution
                break
    
    digit_data = torch.stack(digit_data)
    digit_data = digit_data.view(-1, 784)
    
    with torch.no_grad():
        mu, log_var = model.encode(digit_data)
        mu_mean = mu.mean(dim=0)
        log_var_mean = log_var.mean(dim=0)
        
        # Generate new samples
        samples = []
        for _ in range(num_samples):
            z = model.reparameterize(mu_mean, log_var_mean)
            sample = model.decode(z)
            samples.append(sample.view(28, 28).numpy())
        
        return samples

# Streamlit UI
st.title("Handwritten Digit Generator")
st.write("Generate fake MNIST digits (0-9) using AI")

# User selects a digit
digit = st.selectbox("Choose a digit (0-9):", options=list(range(10)))

if st.button("Generate"):
    st.write(f"## Generated images of digit {digit}")
    
    samples = generate_samples(model, digit)
    
    # Display 5 images in a row
    cols = st.columns(5)
    for i, col in enumerate(cols):
        with col:
            # Convert to image and display
            img_array = (samples[i] * 127.5 + 127.5).astype(np.uint8)  # Scale to 0-255
            img = Image.fromarray(img_array)
            st.image(img, caption=f"Sample {i+1}", width=100)