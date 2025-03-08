import train

if __name__ == "__main__":
    settings = {
        "dataset": "MNIST",

        "print_debug": False,
        "example_image_amount": 8,
        "batch_size": 32,
        "learning_rate": 1e-4, # for Mnsist
        "max_epochs": 100,
        "early_stopping_epochs": 5,

        "model_settings" : {
            "patch_size": 4,
            "embedding_size": 64,
            "attention_heads": 4
        }
    }
    train.train_VIT(settings)