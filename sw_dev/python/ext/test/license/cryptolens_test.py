#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# REF [site] >> https://github.com/Cryptolens/cryptolens-python
def basic_example():
	import licensing, licensing.methods

	# Key verification.

	# REF [site] >> https://app.cryptolens.io/docs/api/v3/QuickStart#api-keys
	RSAPubKey = "<RSAKeyValue><Modulus>sGbvxwdlDbqFXOMlVUnAF5ew0t0WpPW7rFpI5jHQOFkht/326dvh7t74RYeMpjy357NljouhpTLA3a6idnn4j6c3jmPWBkjZndGsPL4Bqm+fwE48nKpGPjkj4q/yzT4tHXBTyvaBjA8bVoCTnu+LiC4XEaLZRThGzIn5KQXKCigg6tQRy0GXE13XYFVz/x1mjFbT9/7dS8p85n8BuwlY5JvuBIQkKhuCNFfrUxBWyu87CFnXWjIupCD2VO/GbxaCvzrRjLZjAngLCMtZbYBALksqGPgTUN7ZM24XbPWyLtKPaXF2i4XRR9u6eTj5BfnLbKAU5PIVfjIS+vNYYogteQ==</Modulus><Exponent>AQAB</Exponent></RSAKeyValue>"
	auth = "WyIyNTU1IiwiRjdZZTB4RmtuTVcrQlNqcSszbmFMMHB3aWFJTlBsWW1Mbm9raVFyRyJd=="

	result = licensing.methods.Key.activate(
		token=auth,
		rsa_pub_key=RSAPubKey,
		product_id=3349,
		key="ICVLD-VVSZR-ZTICT-YKGXL",
		machine_code=licensing.methods.Helpers.GetMachineCode(v=2),
	)

	if result[0] == None or not licensing.methods.Helpers.IsOnRightMachine(result[0], v=2):
		# An error occurred or the key is invalid or it cannot be activated (eg. the limit of activated devices was achieved).
		print(f"The license does not work: {result[1]}.")
		return
	else:
		# Everything went fine if we are here!
		print("The license is valid!")

		license_key = result[0]
		print(f"Feature 1: {license_key.f1}.")
		print(f"License expires: {license_key.expires}.")

	#-----
	# Offline activation (saving/loading licenses).

	if result[0] != None:
		# Save license file to disk.
		with open("./licensefile.skm", "w") as fd:
			fd.write(result[0].save_as_string())

	# Read license file from file.
	with open("./licensefile.skm", "r") as fd:
		license_key = licensing.models.LicenseKey.load_from_string(RSAPubKey, fd.read())
		#license_key = licensing.models.LicenseKey.load_from_string(RSAPubKey, f.read(), 30)  # The maximum number of days. After 30 days, this method will return NoneType.

		if license_key == None or not licensing.methods.Helpers.IsOnRightMachine(license_key, v=2):
			print("NOTE: This license file does not belong to this machine.")
		else:
			print(f"Feature 1: {license_key.f1}.")
			print(f"License expires: {license_key.expires}.")

# REF [site] >> https://github.com/Cryptolens/cryptolens-python
def floating_licenses_example():
	import licensing

	RSAPubKey = "<RSAKeyValue><Modulus>sGbvxwdlDbqFXOMlVUnAF5ew0t0WpPW7rFpI5jHQOFkht/326dvh7t74RYeMpjy357NljouhpTLA3a6idnn4j6c3jmPWBkjZndGsPL4Bqm+fwE48nKpGPjkj4q/yzT4tHXBTyvaBjA8bVoCTnu+LiC4XEaLZRThGzIn5KQXKCigg6tQRy0GXE13XYFVz/x1mjFbT9/7dS8p85n8BuwlY5JvuBIQkKhuCNFfrUxBWyu87CFnXWjIupCD2VO/GbxaCvzrRjLZjAngLCMtZbYBALksqGPgTUN7ZM24XbPWyLtKPaXF2i4XRR9u6eTj5BfnLbKAU5PIVfjIS+vNYYogteQ==</Modulus><Exponent>AQAB</Exponent></RSAKeyValue>"
	auth = "WyIyNTU1IiwiRjdZZTB4RmtuTVcrQlNqcSszbmFMMHB3aWFJTlBsWW1Mbm9raVFyRyJd=="

	result = licensing.methods.Key.activate(
		token=auth,
		rsa_pub_key=RSAPubKey,
		product_id=3349,
		key="ICVLD-VVSZR-ZTICT-YKGXL",
		machine_code=licensing.methods.Helpers.GetMachineCode(v=2),
		floating_time_interval=300,
		max_overdraft=1,
	)

	if result[0] == None or not licensing.methods.Helpers.IsOnRightMachine(result[0], is_floating_license=True, allow_overdraft=True, v=2):
		print(f"An error occurred: {result[1]}.")
	else:
		print("Success")

		license_key = result[0]
		print(f"Feature 1: {license_key.f1}.")
		print(f"License expires: {license_key.expires}.")

# REF [site] >> https://github.com/Cryptolens/cryptolens-python
def trial_key_example():
	import licensing

	trial_key = licensing.methods.Key.create_trial_key(
		"WyIzODQ0IiwiempTRWs4SnBKTTArYUh3WkwyZ0VwQkVyeTlUVkRWK2ZTOS8wcTBmaCJd",
		 3941,
		 licensing.methods.Helpers.GetMachineCode(v=2),
	)
	if trial_key[0] == None:
		print(f"An error occurred: {trial_key[1]}.")
		return

	RSAPubKey = "<RSAKeyValue><Modulus>sGbvxwdlDbqFXOMlVUnAF5ew0t0WpPW7rFpI5jHQOFkht/326dvh7t74RYeMpjy357NljouhpTLA3a6idnn4j6c3jmPWBkjZndGsPL4Bqm+fwE48nKpGPjkj4q/yzT4tHXBTyvaBjA8bVoCTnu+LiC4XEaLZRThGzIn5KQXKCigg6tQRy0GXE13XYFVz/x1mjFbT9/7dS8p85n8BuwlY5JvuBIQkKhuCNFfrUxBWyu87CFnXWjIupCD2VO/GbxaCvzrRjLZjAngLCMtZbYBALksqGPgTUN7ZM24XbPWyLtKPaXF2i4XRR9u6eTj5BfnLbKAU5PIVfjIS+vNYYogteQ==</Modulus><Exponent>AQAB</Exponent></RSAKeyValue>"
	auth = "WyIyNTU1IiwiRjdZZTB4RmtuTVcrQlNqcSszbmFMMHB3aWFJTlBsWW1Mbm9raVFyRyJd=="

	result = licensing.methods.Key.activate(
		token=auth,
		rsa_pub_key=RSAPubKey,
		product_id=3349,
		key=trial_key[0],
		machine_code=licensing.methods.Helpers.GetMachineCode(v=2),
	)

	if result[0] == None or not licensing.methods.Helpers.IsOnRightMachine(result[0], v=2):
		print(f"An error occurred: {result[1]}.")
	else:
		print("Success")

		license_key = result[0]
		print(f"Feature 1: {license_key.f1}.")
		print(f"License expires: {license_key.expires}.")

def main():
	# Install.
	#	pip install licensing

	basic_example()  # Not correctly working.
	floating_licenses_example()  # Not correctly working.
	trial_key_example()  # Not correctly working.

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
