Your Alembic migration goes here (task 4). Create it with `uv run alembic revision -m "create users"` (it lands in this
folder) or write the file by hand. `alembic upgrade head` must create the three tables `users`, `tokens`, `threads`
(columns in users.py); `alembic downgrade base` must drop them again. docker compose runs the upgrade in the `migrate` service.
