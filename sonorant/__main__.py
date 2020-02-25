"""This is the entrypoint for running the Flask service.

You can run the app by running `python -m sonorous` from the root of the repo.
"""


from sonorous.app import PORT, create_app


app = create_app()
app.run(port=PORT)
