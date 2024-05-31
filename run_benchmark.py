

from EasyTSAD.Controller import TSADController
from EasyTSAD.Methods.AR import AR
from EasyTSAD.Evaluations.Protocols import EventF1PA, PointF1PA

if __name__ == "__main__":
    dataset_type = "UTS"
    # datasets = ["AIOPS", "NAB", "TODS", "WSD", "Yahoo", "UCR"]
    datasets = ["AIOPS"]
    
    controller = TSADController('./controller.toml')
    controller.set_dataset(
        dataset_type=dataset_type,
        datasets=datasets,
        dirname="../datasets"
    )
    
    # training_schema must be one of naive, all_in_one, zero_shot
    training_schema = "all_in_one"
    method = "AR"
    config_path = './EasyTSAD/Methods/AR/config.toml'
    
    # step 1
    controller.run_exps(
        method=method,
        training_schema=training_schema,
        cfg_path=config_path,
        do_all=False
    )
    
    # step 2
    # controller.set_evals(
    #     [
    #         PointF1PA(),
    #         EventF1PA(),
    #         EventF1PA(mode="squeeze")
    #     ]
    # )
    # controller.do_evals(
    #     method=method,
    #     training_schema=training_schema
    # )
    
    # step 3
    # controller.plots(
    #     method=method,
    #     training_schema=training_schema
    # )