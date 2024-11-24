from main import main
from ray import tune


num_samples = 10
gpus_per_trial = 0




trainable = tune.with_parameters(
    main,
    num_gpus=gpus_per_trial
)

analysis = tune.run(
    trainable,
    resources_per_trial={
        "cpu": 1,
        "gpu": gpus_per_trial
    },
    metric="loss",
    code="min",
    config=config,
    num_samples=num_samples,
    name="main"
)

print(analysis.best_config)


    

