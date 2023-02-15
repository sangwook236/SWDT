#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import sys, os, tempfile, pickle
import omegaconf
from omegaconf import OmegaConf

# REF [site] >> https://omegaconf.readthedocs.io/en/2.0_branch/usage.html
def usage():
	# Creating.

	conf = OmegaConf.create()
	print(type(conf))
	print(OmegaConf.to_yaml(conf))

	# From a dictionary.
	conf = OmegaConf.create({"k": "v", "list": [1, {"a": "1", "b": "2"}]})
	print(OmegaConf.to_yaml(conf))
	print(conf.list[1].a)
	print(conf["list"][1]["a"])

	# From a list.
	conf = OmegaConf.create([1, {"a": 10, "b": {"a": 10}}])
	print(OmegaConf.to_yaml(conf))

	# From a YAML file.
	'''
	try:
		conf = OmegaConf.load("/path/to/example.yaml")
		print(OmegaConf.to_yaml(conf))
	except FileNotFoundError as ex:
		print(ex)
	'''

	# From a YAML string.
	data = """
# PyTorch Lightning Trainer configuration any argument of the Trainer object can be set here.
trainer:
  devices: 1  # Number of gpus per node.
  accelerator: gpu
  num_nodes: 1  # Number of nodes.
  max_epochs: 10  # How many training epochs to run.
  val_check_interval: 1.0  # Run validation after every epoch.

# Experiment logging configuration.
exp_manager:
  exp_dir: /path/to/my/nemo/experiments
  name: name_of_my_experiment
  create_tensorboard_logger: True
  create_wandb_logger: True
"""

	conf = OmegaConf.create(data)
	print(OmegaConf.to_yaml(conf))

	# From a dot-list.
	dot_list = ["a.aa.aaa=1", "a.aa.bbb=2", "a.bb.aaa=3", "a.bb.bbb=4"]
	conf = OmegaConf.from_dotlist(dot_list)
	print(OmegaConf.to_yaml(conf))

	# From command line arguments.
	# Simulating command line arguments.
	sys.argv = ["your-program.py", "server.port=82", "log.file=log2.txt"]
	conf = OmegaConf.from_cli()
	print(OmegaConf.to_yaml(conf))

	# From structured config.
	from dataclasses import dataclass

	@dataclass
	class MyConfig:
		port: int = 80
		host: str = "localhost"

	# For strict typing purposes, prefer OmegaConf.structured() when creating structured configs.
	conf = OmegaConf.structured(MyConfig)
	print(OmegaConf.to_yaml(conf))

	conf = OmegaConf.structured(MyConfig(port=443))
	print(OmegaConf.to_yaml(conf))

	conf.port = 42  # Ok, type matches.
	conf.port = "1080"  # Ok! "1080" can be converted to an int.
	#conf.port = "oops"  # "oops" cannot be converted to an int.

	#-----
	# Access and manipulation.

	data = """
server:
  port: 80
log:
  file: ???
  rotation: 3600
users:
  - user1
  - user2
"""

	conf = OmegaConf.create(data)

	# Access.
	print(conf.server.port)  # Object style access of dictionary elements.
	print(conf["log"]["rotation"])  # Dictionary style access.
	print(conf.users[0])  # Items in list.

	# Default values.
	# Providing default values.
	#print(conf.missing_key or "a default value")
	print(conf.get("missing_key", "a default value"))

	# Mandatory values.
	#	Use the value ??? to indicate parameters that need to be set prior to access.
	try:
		print(conf.log.file)
	except omegaconf.MissingMandatoryValue as ex:
		print(ex)

	# Manipulation.
	conf.server.port = 81  # Changing existing keys.
	conf.server.hostname = "localhost"  # Adding new keys.
	conf.database = {"hostname": "database01", "port": 3306}  # Adding a new dictionary.

	#-----
	# Serialization.

	# Save/load YAML file.
	conf = OmegaConf.create({"foo": 10, "bar": 20})
	with tempfile.NamedTemporaryFile() as fp:
		OmegaConf.save(config=conf, f=fp.name)
		loaded = OmegaConf.load(fp.name)
		assert conf == loaded

	# Save/load pickle file.
	conf = OmegaConf.create({"foo": 10, "bar": 20})
	with tempfile.TemporaryFile() as fp:
		pickle.dump(conf, fp)
		fp.flush()
		assert fp.seek(0) == 0
		loaded = pickle.load(fp)
		assert conf == loaded

	#-----
	# Variable interpolation.
	#	OmegaConf support variable interpolation, Interpolations are evaluated lazily on access.

	# Config node interpolation.
	#	The interpolated variable can be the dot-path to another node in the configuration, and in that case the value will be the value of that node.
	data = """
server:
  host: localhost
  port: 80

client:
  url: http://${server.host}:${server.port}/
  server_port: ${server.port}
"""

	#conf = OmegaConf.load("/path/to/example.yaml')
	conf = OmegaConf.create(data)
	# Primitive interpolation types are inherited from the referenced value.
	print(conf.client.server_port)
	print(type(conf.client.server_port).__name__)
	# Composite interpolation types are always string.
	print(conf.client.url)
	print(type(conf.client.url).__name__)

	# Environment variable interpolation.
	data = """
user:
  name: ${env:USER}
  home: /home/${env:USER}
"""

	#conf = OmegaConf.load("/path/to/example.yaml")
	conf = OmegaConf.create(data)
	try:
		print(conf.user.name)
		print(conf.user.home)
	except omegaconf.errors.UnsupportedInterpolationType as ex:
		print(ex)

	# You can specify a default value to use in case the environment variable is not defined.
	# The following example sets 12345 as the the default value for the DB_PASSWORD environment variable.
	cfg = OmegaConf.create({
		"database": {"password": "${env:DB_PASSWORD,12345}"}
	})
	try:
		print(cfg.database.password)
		OmegaConf.clear_cache(cfg)  # Clear resolver cache.
		os.environ["DB_PASSWORD"] = "secret"
		print(cfg.database.password)
	except omegaconf.errors.UnsupportedInterpolationType as ex:
		print(ex)

	# Custom interpolations.
	#OmegaConf.register_resolver("plus_10", lambda x: int(x) + 10)  # Deprecated.
	OmegaConf.register_new_resolver("plus_10", lambda x: int(x) + 10)
	c = OmegaConf.create({"key": "${plus_10:990}"})
	print(c.key)

	#OmegaConf.register_resolver("concat", lambda x, y: x + y)  # Deprecated.
	OmegaConf.register_new_resolver("concat", lambda x, y: x + y)
	c = OmegaConf.create({
		"key1": "${concat:Hello,World}",
		"key_trimmed": "${concat:Hello , World}",
		"escape_whitespace": "${concat:Hello,\ World}",
	})
	print(c.key1)
	print(c.key_trimmed)
	print(c.escape_whitespace)

	#-----
	# Merging configurations.

	# Simulate command line arguments.
	sys.argv = ["program.py", "server.port=82"]

	data1 = """
server:
  port: 80
users:
  - user1
  - user2
"""
	data2 = """
log:
  file: log.txt
"""

	#base_conf = OmegaConf.load("/path/to/example1.yaml")
	base_conf = OmegaConf.create(data1)
	#second_conf = OmegaConf.load("/path/to/example2.yaml")
	second_conf = OmegaConf.create(data2)
	cli_conf = OmegaConf.from_cli()

	# Merge them all.
	conf = OmegaConf.merge(base_conf, second_conf, cli_conf)
	print(OmegaConf.to_yaml(conf))

	#-----
	# Configuration flags.

	# Read-only flag.
	conf = OmegaConf.create({"a": {"b": 10}})
	OmegaConf.set_readonly(conf, True)
	try:
		conf.a.b = 20
	except omegaconf.ReadonlyConfigError as ex:
		print(ex)

	# You can temporarily remove the read only flag from a config object.
	conf = OmegaConf.create({"a": {"b": 10}})
	OmegaConf.set_readonly(conf, True)
	with omegaconf.read_write(conf):
		conf.a.b = 20
	print(conf.a.b)

	# Struct flag.
	#	By default, OmegaConf dictionaries allow read and write access to unknown fields.
	#	If a field does not exist, accessing it will return None and writing it will create the field.
	#	It's sometime useful to change this behavior.
	conf = OmegaConf.create({"a": {"aa": 10, "bb": 20}})
	OmegaConf.set_struct(conf, True)
	try:
		conf.a.cc = 30
	except omegaconf.errors.ConfigAttributeError as ex:
		print(ex)

	# You can temporarily remove the struct flag from a config object.
	conf = OmegaConf.create({"a": {"aa": 10, "bb": 20}})
	OmegaConf.set_struct(conf, True)
	with omegaconf.open_dict(conf):
		conf.a.cc = 30
	print(conf.a.cc)

	#-----
	# Utility functions.

	# Tests if a value is missing ('???').
	cfg = OmegaConf.create({
		"foo" : 10,
		"bar": "???"
	})
	assert not OmegaConf.is_missing(cfg, "foo")
	assert OmegaConf.is_missing(cfg, "bar")

	# Tests if a value is an interpolation.
	cfg = OmegaConf.create({
		"foo" : 10,
		"bar": "${foo}"
	})
	assert not OmegaConf.is_interpolation(cfg, "foo")
	assert OmegaConf.is_interpolation(cfg, "bar")

	# Tests if a value is None.
	'''
	cfg = OmegaConf.create({
		"foo" : 10,
		"bar": None,
	})
	assert not OmegaConf.is_none(cfg, "foo")
	assert OmegaConf.is_none(cfg, "bar")
	# Missing keys are interpreted as None.
	assert OmegaConf.is_none(cfg, "no_such_key")
	'''

	# Tests if an object is an OmegaConf object, or if it's representing a list or a dict.
	# Dict:
	d = OmegaConf.create({"foo": "bar"})
	assert OmegaConf.is_config(d)
	assert OmegaConf.is_dict(d)
	assert not OmegaConf.is_list(d)
	# List:
	l = OmegaConf.create([1, 2, 3])
	assert OmegaConf.is_config(l)
	assert OmegaConf.is_list(l)
	assert not OmegaConf.is_dict(l)

	# OmegaConf config objects looks very similar to python dict and list, but in fact are not.
	# Use OmegaConf.to_container(cfg : Container, resolve : bool) to convert to a primitive container.
	conf = OmegaConf.create({"foo": "bar", "foo2": "${foo}"})
	assert type(conf) == omegaconf.DictConfig
	primitive = OmegaConf.to_container(conf)
	assert type(primitive) == dict
	print(primitive)
	resolved = OmegaConf.to_container(conf, resolve=True)
	print(resolved)

	# OmegaConf.select() allow you to select a config node or value using a dot-notation key.
	cfg = OmegaConf.create({
		"foo" : {
			"bar": {
				"zonk" : 10,
				"missing" : "???"
			}
		}
	})
	assert OmegaConf.select(cfg, "foo") == {
		"bar":  {
			"zonk" : 10,
			"missing" : "???"
		}
	}
	assert OmegaConf.select(cfg, "foo.bar") == {
		"zonk" : 10,
		"missing" : "???"
	}
	assert OmegaConf.select(cfg, "foo.bar.zonk") == 10
	assert OmegaConf.select(cfg, "foo.bar.missing") is None
	try:
		OmegaConf.select(cfg,
			"foo.bar.missing",
			throw_on_missing=True
		)
	except omegaconf.errors.MissingMandatoryValue as ex:
		print(ex)

	# OmegaConf.update() allow you to update values in your config using a dot-notation key.
	cfg = OmegaConf.create({"foo" : {"bar": 10}})
	OmegaConf.update(cfg, "foo.bar", 20, merge=True)  # Merge has no effect because the value is a primitive.
	assert cfg.foo.bar == 20
	OmegaConf.update(cfg, "foo.bar", {"zonk" : 30}, merge=False)  # Set.
	assert cfg.foo.bar == {"zonk" : 30}
	OmegaConf.update(cfg, "foo.bar", {"oompa" : 40}, merge=True)  # Merge.
	assert cfg.foo.bar == {"zonk" : 30, "oompa" : 40}

	# Creates a copy of a DictConfig that contains only specific keys.
	conf = OmegaConf.create({"a": {"b": 10}, "c":20})
	print(OmegaConf.to_yaml(conf))

	c = OmegaConf.masked_copy(conf, ["a"])
	print(OmegaConf.to_yaml(c))

def main():
	usage()

#---------------------------------------------------------------------

if "__main__" == __name__:
	main()
