# Prepares the app for deployment.

# Plik zawiera kod do wdrażania aplikacji, w tym:

# Tworzenie katalogu deploymentu,
# Kopiowanie wytrenowanego modelu,
# Budowanie obrazu Docker,
# Uruchamianie kontenera aplikacji na porcie 8080.

import os
import shutil
import subprocess

def create_deployment_directory(output_dir):
    """Creates a clean deployment directory."""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

def copy_model_files(model_source, model_dest):
    """Copies trained model files to the deployment directory."""
    if not os.path.exists(model_source):
        raise FileNotFoundError("Model file not found!")
    shutil.copy(model_source, model_dest)

def build_docker_image(image_name):
    """Builds a Docker image for deployment."""
    subprocess.run(["docker", "build", "-t", image_name, "./"], check=True)

def run_docker_container(image_name, container_name, port):
    """Runs the application in a Docker container."""
    subprocess.run([
        "docker", "run", "-d", "--name", container_name, "-p", f"{port}:5000", image_name
    ], check=True)

def deploy_application():
    """Main function to handle the deployment process."""
    output_dir = "./deployment"
    model_source = "./models/alzheimer_cnn.h5"
    model_dest = os.path.join(output_dir, "model.h5")
    image_name = "alzheimer-voice-api"
    container_name = "alzheimer_app"
    port = 8080
    
    print("Setting up deployment directory...")
    create_deployment_directory(output_dir)
    
    print("Copying model files...")
    copy_model_files(model_source, model_dest)
    
    print("Building Docker image...")
    build_docker_image(image_name)
    
    print("Running Docker container...")
    run_docker_container(image_name, container_name, port)
    
    print(f"Deployment successful! The app is running on port {port}.")

if __name__ == "__main__":
    deploy_application()
