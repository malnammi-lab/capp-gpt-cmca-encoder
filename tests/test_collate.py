"""Padding contract. Getting this wrong trains the model on fabricated faces."""

import torch

from src.data.collate import collate_cmca, collate_fn_brep


def test_alias_is_the_same_function():
    assert collate_fn_brep is collate_cmca


def test_batch_keys(batch):
    expected = {"faces_uv", "faces_attr", "edges_attr", "bias", "adjacency",
                "attr_similarity", "semantic_labels", "face_mask", "edge_mask",
                "instance_labels", "bottom_labels", "model_ids"}
    assert set(batch) == expected


def test_everything_is_padded_to_the_batch_max(dataset):
    items = [dataset[i] for i in range(3)]
    b = collate_cmca(items)
    max_f = max(it["faces_uv"].shape[0] for it in items)
    max_e = max(it["edges_attr"].shape[0] for it in items)

    assert b["faces_uv"].shape[:2] == (3, max_f)
    assert b["faces_attr"].shape == (3, max_f, dataset.attr_dim)
    assert b["edges_attr"].shape[:2] == (3, max_e)
    for k in ("bias", "adjacency", "attr_similarity", "instance_labels"):
        assert b[k].shape == (3, max_f, max_f), k
    assert b["semantic_labels"].shape == (3, max_f)
    assert b["face_mask"].shape == (3, max_f)
    assert b["edge_mask"].shape == (3, max_e)


def test_masks_mark_exactly_the_real_entries(dataset):
    items = [dataset[i] for i in range(3)]
    b = collate_cmca(items)
    for i, it in enumerate(items):
        assert b["face_mask"][i].sum().item() == it["faces_uv"].shape[0]
        assert b["edge_mask"][i].sum().item() == it["edges_attr"].shape[0]
        # and they are a prefix, not scattered
        nf = it["faces_uv"].shape[0]
        assert b["face_mask"][i][:nf].all() and not b["face_mask"][i][nf:].any()


def test_padded_labels_are_minus_one_not_zero(dataset):
    """Class 0 is real ("Chamfer" in the MFInstSeg map). Padding with 0 meant a
    consumer that forgot the mask trained on fabricated chamfers, silently. -1
    fails loudly instead."""
    items = [dataset[i] for i in range(len(dataset))]
    b = collate_cmca(items)
    padded = b["semantic_labels"][~b["face_mask"]]
    if padded.numel():
        assert (padded == -1).all()
    assert (b["semantic_labels"][b["face_mask"]] >= 0).all()


def test_real_content_survives_padding(dataset):
    items = [dataset[i] for i in range(3)]
    b = collate_cmca(items)
    for i, it in enumerate(items):
        nf = it["faces_uv"].shape[0]
        assert torch.allclose(b["faces_attr"][i, :nf], it["faces_attr"])
        assert torch.allclose(b["adjacency"][i, :nf, :nf], it["adjacency"])
        assert torch.equal(b["semantic_labels"][i, :nf], it["semantic_labels"])


def test_padding_region_is_zero(dataset):
    items = [dataset[i] for i in range(3)]
    b = collate_cmca(items)
    for i, it in enumerate(items):
        nf = it["faces_uv"].shape[0]
        assert b["adjacency"][i, nf:, :].abs().sum() == 0
        assert b["faces_attr"][i, nf:].abs().sum() == 0


def test_model_ids_are_carried_through(dataset):
    items = [dataset[i] for i in range(3)]
    b = collate_cmca(items)
    assert b["model_ids"] == [it["model_id"] for it in items]
    assert isinstance(b["model_ids"], list), "must stay a list, not become a tensor"


def test_singleton_batch(dataset):
    b = collate_cmca([dataset[0]])
    assert b["faces_uv"].shape[0] == 1
    assert b["face_mask"].all(), "nothing to pad in a batch of one"
