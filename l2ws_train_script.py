import sys
import examples.markowitz as markowitz
import examples.osc_mass as osc_mass
import examples.vehicle as vehicle
import hydra
import pdb
import yaml
from utils.date_utils import copy_data_file


@hydra.main(config_path='configs/markowitz', config_name='markowitz_run.yaml')
def main_run_markowitz(cfg):
    example = 'markowitz'
    markowitz.run(cfg)


@hydra.main(config_path='configs/osc_mass', config_name='osc_mass_run.yaml')
def main_run_osc_mass(cfg):
    example = 'osc_mass'
    agg_datetime = cfg.data.datetime
    copy_data_file(example, agg_datetime)
    osc_mass.run(cfg)


@hydra.main(config_path='configs/vehicle', config_name='vehicle_run.yaml')
def main_run_vehicle(cfg):
    example = 'vehicle'
    agg_datetime = cfg.data.datetime
    copy_data_file(example, agg_datetime)
    vehicle.run(cfg)


if __name__ == '__main__':
    if sys.argv[2] == 'cluster':
        base = 'hydra.run.dir=/scratch/gpfs/rajivs/learn2warmstart/outputs/'
    elif sys.argv[2] == 'local':
        base = 'hydra.run.dir=outputs/'
    if sys.argv[1] == 'markowitz':
        # step 1. remove the markowitz argument -- otherwise hydra uses it as an override
        # step 2. add the train_outputs/... argument for train_outputs not outputs
        # sys.argv[1] = 'hydra.run.dir=outputs/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv[1] = base + 'markowitz/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_markowitz()
    elif sys.argv[1] == 'osc_mass':
        sys.argv[1] = base + 'osc_mass/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_osc_mass()
    elif sys.argv[1] == 'vehicle':
        sys.argv[1] = base + 'vehicle/train_outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}'
        sys.argv = [sys.argv[0], sys.argv[1]]
        main_run_vehicle()
