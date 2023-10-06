#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import discord

# REF [site] >> https://docs.pycord.dev/en/stable/quickstart.html
def minimal_bot_quickstart():
	intents = discord.Intents.default()
	intents.message_content = True

	client = discord.Client(intents=intents)

	@client.event
	async def on_ready():
		print(f"We have logged in as {client.user}")

	@client.event
	async def on_message(message):
		if message.author == client.user:
			return

		if message.content.startswith("$hello"):
			await message.channel.send("Hello!")

	client.run("your token here")

# REF [site] >> https://docs.pycord.dev/en/stable/quickstart.html
def minimal_bot_with_slash_commands_quickstart():
	bot = discord.Bot()

	@bot.event
	async def on_ready():
		print(f"We have logged in as {bot.user}")

	@bot.slash_command(guild_ids=[your, guild_ids, here])
	async def hello(ctx):
		await ctx.respond("Hello!")

	bot.run("your token here")

def main():
	minimal_bot_quickstart()  # Not yet tested.
	minimal_bot_with_slash_commands_quickstart()  # Not yet tested.

#--------------------------------------------------------------------

if "__main__" == __name__:
	main()
