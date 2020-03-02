"""Interface to Amazon Polly TTS."""

import base64
import os
from contextlib import closing
from typing import Tuple

import boto3

# Optional TODO: read these in while creating the app
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
REGION_NAME = "us-west-2"

POLLY_CLIENT = boto3.Session(
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, region_name=REGION_NAME
).client("polly")


def get_audio(pronunciation: Tuple[str, ...]) -> str:
    pronunciation_string = "".join(pronunciation)
    speech = f"<phoneme alphabet='ipa' ph='{pronunciation_string}'></phoneme>"
    response = POLLY_CLIENT.synthesize_speech(
        Text=speech, TextType="ssml", VoiceId="Joanna", OutputFormat="mp3",
    )

    with closing(response["AudioStream"]) as stream:
        audio = base64.encodebytes(stream.read())

    return audio.decode("ascii")
