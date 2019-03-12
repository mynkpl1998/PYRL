import yaml

def ReadExpConfig(config_file):
	with open(config_file, "r") as handle:
		file = yaml.load(handle)

	return file