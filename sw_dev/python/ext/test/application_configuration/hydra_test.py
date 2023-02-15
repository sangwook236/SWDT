#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import hydra
from omegaconf import DictConfig, OmegaConf

# REF [site] >> https://hydra.cc/docs/intro/
@hydra.main(version_base=None, config_path="conf", config_name="config")
def quick_start_guide(cfg : DictConfig) -> None:
	print(OmegaConf.to_yaml(cfg))

def main():
	# Basic example:
	#	conf/config.yaml is loaded automatically when you run your application.
	#		python hydra_test.py
	#		python hydra_test.py --config-path conf2 --config-name config2
	#	You can override values in the loaded config from the command line.
	#		python hydra_test.py db.user=root db.pass=1234

	# Composition example:
	#	defaults is a special directive telling Hydra to use db/mysql.yaml when composing the configuration object.
	#	The resulting cfg object is a composition of configs from defaults with configs specified in your config.yaml.
	#		python hydra_test.py
	#		python hydra_test.py db=postgresql db.timeout=20

	# Multirun:
	#	python hydra_test.py --multirun db=mysql,postgresql

	quick_start_guide()

"""
# REF [site] >> https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/
@hydra.main(version_base=None)
def simple_cli_tutorial(cfg : DictConfig) -> None:
	print(OmegaConf.to_yaml(cfg))

def main():
	# You can add config values via the command line.
	# The + indicates that the field is new.
	#	python hydra_test.py +db.driver=mysql +db.user=omry +db.password=secret

	simple_cli_tutorial()
"""

#---------------------------------------------------------------------

if "__main__" == __name__:
	main()
