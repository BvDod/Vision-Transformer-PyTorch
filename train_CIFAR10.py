import train

if __name__ == "__main__":
    settings = {
        "dataset": "CIFAR",

        "print_debug": False,
        "example_image_amount": 8,
        "batch_size": 128,
        "learning_rate": 3e-4, # for Mnsist
        "max_epochs": 300,
        "early_stopping_epochs": 50,
        "enable_augmentations": True,

        "model_settings" : {
            "patch_size": 4,
            "embedding_size": 256,
            "attention_heads": 8,
            "transformer_layers": 6
        }
    }
    train.train_VIT(settings)