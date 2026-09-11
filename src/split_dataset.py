import random
import pandas as pd

INPUT_FILE = "data/dataset.csv"

TRAIN_FILE = "data/train.csv"
VAL_FILE = "data/val.csv"
TEST_FILE = "data/test.csv"

SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def split_groups(df, groups, train_ratio, val_ratio, test_ratio):
    """
    Split a dataframe's groups while keeping every group intact.

    Groups are assigned greedily by group size, trying to keep
    each split close to the requested number of files.
    """

    rng = random.Random(SEED)

    group_sizes = (
        df.groupby("group")
        .size()
        .to_dict()
    )

    group_list = list(groups)

    # Shuffle first so equal-sized groups don't always go
    # to the same split.
    rng.shuffle(group_list)

    # Largest groups first.
    group_list.sort(
        key=lambda g: group_sizes[g],
        reverse=True,
    )

    total = sum(group_sizes.values())

    targets = {
        "train": total * train_ratio,
        "val": total * val_ratio,
        "test": total * test_ratio,
    }

    assigned = {
        "train": [],
        "val": [],
        "test": [],
    }

    counts = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    # Assign large groups first to whichever split is
    # currently furthest below its target.
    for group in group_list:

        size = group_sizes[group]

        scores = {}

        for split in ("train", "val", "test"):
            target = targets[split]
            current = counts[split]

            # How far below target are we?
            deficit = target - current

            # Prefer splits with the largest deficit.
            scores[split] = deficit

        selected = max(
            scores,
            key=scores.get,
        )

        assigned[selected].append(group)
        counts[selected] += size

    return assigned


def make_split(df, groups):
    return df[df["group"].isin(groups)].copy()


def print_stats(name, df):

    print(f"\n{'=' * 60}")
    print(name)
    print(f"{'=' * 60}")

    print(f"Files:  {len(df)}")
    print(f"Groups: {df['group'].nunique()}")

    print("\nLabels:")
    print(
        df["label"]
        .value_counts()
        .sort_index()
    )

    print("\nSources:")
    print(
        df["source"]
        .value_counts()
    )

    print("\nGenerators:")
    print(
        df["generator"]
        .replace("", "unknown")
        .value_counts()
    )


def verify_no_overlap(train, val, test):

    train_groups = set(train["group"])
    val_groups = set(val["group"])
    test_groups = set(test["group"])

    train_val = train_groups & val_groups
    train_test = train_groups & test_groups
    val_test = val_groups & test_groups

    if train_val:
        raise RuntimeError(
            f"Train/val group leakage: {train_val}"
        )

    if train_test:
        raise RuntimeError(
            f"Train/test group leakage: {train_test}"
        )

    if val_test:
        raise RuntimeError(
            f"Val/test group leakage: {val_test}"
        )

    print("\nNo group leakage detected.")


def main():

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "file",
        "label",
        "source",
        "speaker",
        "generator",
        "language",
        "group",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    # ---------------------------------------------------------
    # We deliberately split each dataset/provenance category
    # separately.
    #
    # This prevents the fact that GaryStafford contains very
    # large groups from dominating the entire split.
    # ---------------------------------------------------------

    assignments = {
        "train": [],
        "val": [],
        "test": [],
    }

    # ---------------------------------------------------------
    # Split human data by source
    # ---------------------------------------------------------

    human = df[df["label"] == 0]

    for source in human["source"].unique():

        source_df = human[
            human["source"] == source
        ]

        groups = source_df["group"].unique()

        result = split_groups(
            source_df,
            groups,
            TRAIN_RATIO,
            VAL_RATIO,
            TEST_RATIO,
        )

        assignments["train"].extend(result["train"])
        assignments["val"].extend(result["val"])
        assignments["test"].extend(result["test"])

    # ---------------------------------------------------------
    # Split AI data by generator
    # ---------------------------------------------------------

    ai = df[df["label"] == 1]

    for generator in ai["generator"].unique():

        generator_df = ai[
            ai["generator"] == generator
        ]

        groups = generator_df["group"].unique()

        result = split_groups(
            generator_df,
            groups,
            TRAIN_RATIO,
            VAL_RATIO,
            TEST_RATIO,
        )

        assignments["train"].extend(result["train"])
        assignments["val"].extend(result["val"])
        assignments["test"].extend(result["test"])

    # ---------------------------------------------------------
    # Create dataframes
    # ---------------------------------------------------------

    train_df = make_split(
        df,
        assignments["train"],
    )

    val_df = make_split(
        df,
        assignments["val"],
    )

    test_df = make_split(
        df,
        assignments["test"],
    )

    # ---------------------------------------------------------
    # Shuffle rows
    # ---------------------------------------------------------

    train_df = train_df.sample(
        frac=1,
        random_state=SEED,
    ).reset_index(drop=True)

    val_df = val_df.sample(
        frac=1,
        random_state=SEED,
    ).reset_index(drop=True)

    test_df = test_df.sample(
        frac=1,
        random_state=SEED,
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Verify
    # ---------------------------------------------------------

    verify_no_overlap(
        train_df,
        val_df,
        test_df,
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    train_df.to_csv(
        TRAIN_FILE,
        index=False,
    )

    val_df.to_csv(
        VAL_FILE,
        index=False,
    )

    test_df.to_csv(
        TEST_FILE,
        index=False,
    )

    # ---------------------------------------------------------
    # Print statistics
    # ---------------------------------------------------------

    print_stats(
        "TRAIN",
        train_df,
    )

    print_stats(
        "VALIDATION",
        val_df,
    )

    print_stats(
        "TEST",
        test_df,
    )

    print("\nFiles written:")
    print(f"  {TRAIN_FILE}")
    print(f"  {VAL_FILE}")
    print(f"  {TEST_FILE}")


if __name__ == "__main__":
    main()