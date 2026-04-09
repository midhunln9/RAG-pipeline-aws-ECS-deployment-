import uuid
import random

from locust import HttpUser, task, between

QUERIES = [
    "What is financial compliance and why is it important?",
    "Explain the key principles of risk management.",
    "What are the regulatory requirements for banking?",
    "How does anti-money laundering work?",
    "What is the role of internal audit?",
    "Describe the Basel III framework.",
    "What are the penalties for non-compliance?",
    "How do companies manage operational risk?",
]


class FastAPIUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        """Each simulated user gets a unique session_id."""
        self.session_id = str(uuid.uuid4())

    @task
    def ask_endpoint(self):
        payload = {
            "query": random.choice(QUERIES),
            "session_id": self.session_id,
        }

        self.client.post(
            "/ask",
            json=payload,
            headers={"Content-Type": "application/json"},
            name="POST /ask",
        )
