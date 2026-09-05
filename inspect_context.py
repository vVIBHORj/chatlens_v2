from chat_database import get_message_range


def show(message_id, before=15, after=15):
    print("=" * 80)
    print(f"CONTEXT AROUND MESSAGE {message_id}")
    print("=" * 80)

    rows = get_message_range(
        start_id=max(0, message_id - before),
        end_id=message_id + after,
    )

    for row in rows:
        print(
            f"[{row['id']}] "
            f"{row['timestamp']} "
            f"{row['sender']}: "
            f"{row['message']}"
        )


if __name__ == "__main__":

    # Laptop hits from the diagnostic
    show(286)

    show(167)

    # Udit's strongest semantic hit
    show(924)