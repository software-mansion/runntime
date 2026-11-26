"""Privacy Filter reference: the spans opf finds in each text, and the logits
of its model for the text's token ids, in f32 on the CPU. The logits come
twice: on the upstream weights, and on the weights of the file runntime
hosts, its int8 experts turned back into floats.

    uv run python -m runntime_parity.privacy_filter

opf is the upstream code, github.com/openai/privacy-filter at the commit
pyproject.toml pins. It runs the `original/` checkpoint of the repo, which
is bf16: the model's weights are cast to f32 before any text is run.
"""

import json
import os
from importlib.metadata import distribution
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from numpy.typing import NDArray
from opf import OPF
from safetensors.torch import load as load_tensors

from .common import checked, cosine_noise, noise, save

REPO = "openai/privacy-filter"
REVISION = "7ffa9a043d54d1be65afb281eddf0ffbe629385b"
# The commit of opf in pyproject.toml.
OPF_COMMIT = "f7f00ca7fb869683eb732c010299d901457f19c3"
# The files of `original/` opf reads, each with its sha256.
CHECKPOINT_SHA256 = {
    "config.json": "048a20604a3622de208d30df57cd5424bb583639b9ba20ddd7da593d3f89a248",
    "model.safetensors": "9c262cbe68a0c8a50590a648ef8341a2b7d3be1fa11dfb79893fe0b03ce57b5c",
    "viterbi_calibration.json": "bbc8611ef08a55ed72d64856cbbbb9a91db8dfa881f0a92e2afbad6e4bbc775a",
}

# The file runntime hosts: the upstream weights with the experts' in int8.
HOSTED_REPO = "software-mansion/runntime-privacy-filter"
HOSTED_REVISION = "74a8a921d1841f3e8890d640cb5e5ee73775185f"
HOSTED_FILE = "base/int8g640/model.safetensors"
HOSTED_SHA256 = "1dbd74f7bd7d01269cf19ede73e9ac8a104abd47b03360735b29d7624621045f"

# Settings of opf that change its math or its code path.
REFUSED_SETTINGS = ("OPF_ATTN_LOW_PRECISION", "OPF_TORCH_COMPILE")

# Texts written for the test. Every person, address, number and secret in
# them is made up. One has CRLF line ends, one no-break spaces.
TEXTS = [
    "Contact Jane Doe at jane.doe@example.com.",
    "Please call Marcus Oyelaran on 415-555-0137 before Friday, 14 March 2025.",
    "Ship the order to 742 Evergreen Terrace, Springfield, OR 97477.",
    "The invoice goes to account 4000 1234 5678 9010, routing number 110000000.",
    "Reset your password at https://portal.example.org/reset?user=mlindqvist&step=2 today.",
    "The staging database password is Maple-Otter-7291! and the API key is tok_demo_9f8a7b6c5d4e3f2a1b0c.",
    "The library opens at nine and closes at five. Bring a pen, a notebook and something to drink.",
    "Name: Priya Raghunathan\nEmail: priya.r@example.net\nPhone: +44 20 7946 0958\nDOB: 1988-07-23",
    "Amelia Brandt, amelia@brandt.example, 030 5550 1234, Lindenstraße 14, 10969 Berlin",
    "Zamówienie odbierze Zofia Wróblewska, tel. 601 555 204, ul. Długa 7/3, 31-147 Kraków. Urodzona 3 maja 1991 r.",
    "Écrivez à Hélène Lefèvre (helene.lefevre@exemple.fr) avant le 2 février 2026\u00a0; "
    "son numéro est le 01\u00a055\u00a055\u00a001\u00a042.",
    "El señor Iñaki Etxeberria vive en Calle de la Luna 28, 3.º B, 28004 Madrid, "
    "y su IBAN es ES91 2100 0418 4502 0005 1332.",
    "山田太郎さんの電話番号は03-5555-0199、メールはtaro.yamada@example.jpです。",
    "🎉 Happy birthday Tomás! 🎂 See you on 9 August at 18 Rue des Lilas 🏡 — "
    "text me on 555-0144 📱 or mail tomas.ribeiro@example.pt 💌",
    "export DB_PASSWORD='c0rrect-Horse-battery-88'\n"
    "curl -H 'Authorization: Bearer 7c9e6679f7425d0a8c3b1e2d4f5a6b7c' https://api.example.com/v1/users/8841",
    "My profile is https://social.example.com/@noor.haddad and my portfolio lives at www.noorhaddad.example/work.",
    "Born on 12/04/1979, hired 1 September 2004, retired on the 30th of June 2021.",
    "Customer ID 88-204-771, passport X1234567, SSN 078-05-1120, card 5555 5555 5555 4444 exp 09/27.",
    "From: Oluwaseun Adebayo <seun.adebayo@example.ng>\r\nTo: Li Wei <li.wei@example.cn>\r\n"
    "Date: Tue, 4 Feb 2025 09:14\r\n\r\nHi Li, my new number is +234 803 555 0167. Regards, Seun",
    "name\temail\tphone\nGreta Holm\tgreta.holm@example.se\t070-555 01 23\n"
    "Anders Vik\tanders@vik.example\t070-555 01 87",
    '{"user": "kwame.mensah", "full_name": "Kwame Mensah", "address": "12 Independence Ave, Accra", '
    '"token": "a1b2c3d4e5f60718293a4b5c6d7e8f90"}',
    "hi my name is dana, i cant log in. my email is dana_k92@example.com and the code you sent was 481516",
    "Ada Lovelace wrote the first program in 1843. My neighbour Ada Kowalczyk, born 1990, "
    "lives at 5 Mill Lane, Leeds LS1 4AP.",
    "Dear Ms. Ingrid Solberg,\n\n"
    "Thank you for your letter of 3 November 2025 about the flat at 27 Harbour Row, Apartment 4C, Bergen. "
    "We have looked at the points you raised and would like to answer them one by one.\n\n"
    "First, the heating. The building's boiler was replaced last winter, and the radiators in the "
    "bedrooms were bled at the same time. If a radiator still stays cold, the valve is most likely "
    "stuck, which a plumber can free in a few minutes. We have asked our caretaker to visit you. "
    "He will ring the bell twice, so that you know it is him and not a salesman.\n\n"
    "Second, the noise from the courtyard. The café on the ground floor has agreed to close its "
    "terrace at ten in the evening on weekdays and at eleven at weekends. The delivery vans will no "
    "longer come before seven in the morning. Should that change, please tell us, and we will talk "
    "to the owner again.\n\n"
    "Third, the storage room in the cellar. The room marked with your flat's number is yours to use. "
    "The padlock on it was left by the previous tenant, and nobody has its key, so the caretaker will "
    "cut it off and fit a new one. The cellar is dry, but we advise against keeping books or clothes "
    "directly on the floor.\n\n"
    "Finally, the deposit. It is held in a separate account, as the law asks, and will be returned "
    "within four weeks of the day you move out, less the cost of any damage beyond normal wear. "
    "Interest on it is yours.\n\n"
    "The caretaker, Henrik Aasen, will come on 12 November 2025 between nine and noon. You can reach "
    "him on +47 55 55 01 29 or at henrik.aasen@harbourrow.example. If the time does not suit you, "
    "write to me at lena.dahl@harbourrow.example and we will find another.\n\n"
    "Kind regards,\nLena Dahl\nHarbour Row Lettings",
    "Patient: Rahul Deshmukh, MRN 00482913, seen 2025-01-17 by Dr. Elena Sokolova. Follow-up phone: (212) 555-0190.",
    "The Wi-Fi password is sunflower-Quartz-42 and the router admin page is http://192.168.1.1/admin.",
    "Thanks, Bob",
    "Sayın Ayşe Yıldız, randevunuz 15 Ekim 2025 tarihinde; adres: Atatürk Cad. No: 45, Kadıköy, İstanbul.",  # noqa: RUF001
]


def checkpoint() -> Path:
    """The folder of the `original/` checkpoint, each file opf reads checked
    against its sha256."""
    root = Path(snapshot_download(REPO, revision=REVISION, allow_patterns=["original/*"])) / "original"
    for name, sha256 in CHECKPOINT_SHA256.items():
        checked(str(root / name), (root / name).read_bytes(), sha256)
    return root


def dequantized(words: torch.Tensor, scales: torch.Tensor, experts: int) -> torch.Tensor:
    """The f32 weight [experts, inputs, outputs] of int8 values packed in
    `words`, as runntime's format lays them out: little-endian in tiles of
    16, one per expert, 4 outputs and 4 inputs, tiles by expert, outputs,
    then inputs, the values of a tile by output, then input. `scales` holds
    one scale per expert, group of inputs and output, [experts · groups,
    outputs]."""
    outputs = scales.shape[1]
    inputs = words.numel() * 4 // (experts * outputs)
    tiles = torch.from_numpy(words.numpy().view(np.int8)).reshape(experts, outputs // 4, inputs // 4, 4, 4)
    ints = tiles.permute(0, 2, 4, 1, 3).reshape(experts, inputs, outputs)
    groups = scales.reshape(experts, -1, outputs)
    return ints.float() * groups.repeat_interleave(inputs // groups.shape[1], dim=1)


def load_hosted(model: torch.nn.Module) -> None:
    """Gives `model` the weights of the hosted file, all in f32."""
    path = hf_hub_download(HOSTED_REPO, HOSTED_FILE, revision=HOSTED_REVISION)
    file = load_tensors(checked(path, Path(path).read_bytes(), HOSTED_SHA256))
    experts = file["block.0.mlp.gate.bias"].shape[0]

    def weight(name: str) -> torch.Tensor:
        return dequantized(file[f"{name}Weight"], file[f"{name}Scales"], experts)

    for name, parameter in model.named_parameters():
        block = name.rsplit(".", 1)[0]
        if name.endswith("mlp1_weight"):
            value = torch.cat([weight(f"{block}.glu"), weight(f"{block}.lin")], dim=-1)
        elif name.endswith("mlp1_bias"):
            value = torch.cat([file[f"{block}.gluBias"], file[f"{block}.linBias"]], dim=-1)
        elif name.endswith("mlp2_weight"):
            value = weight(f"{block}.out")
        elif name.endswith("mlp2_bias"):
            value = file[f"{block}.outBias"]
        else:
            value = file[name]
        if value.shape != parameter.shape:
            raise RuntimeError(f"{name} is {tuple(value.shape)} in the hosted file, {tuple(parameter.shape)} in opf")
        parameter.data = value.float()


def logits(model: torch.nn.Module, ids: list[int]) -> NDArray:
    """The model's logits for one text's token ids, [tokens, labels], as opf
    asks for them."""
    tokens = torch.tensor([ids], dtype=torch.int32)
    return model(tokens, attention_mask=torch.ones_like(tokens, dtype=torch.bool))[0].numpy()


def utf16(text: str, index: int) -> int:
    """The string index of a character of `text` as JavaScript counts it."""
    return len(text[:index].encode("utf-16-le")) // 2


def main() -> None:
    installed = json.loads(distribution("opf").read_text("direct_url.json"))["vcs_info"]["commit_id"]
    if installed != OPF_COMMIT:
        raise RuntimeError(f"opf is installed from commit {installed}, expected {OPF_COMMIT}")
    for name in REFUSED_SETTINGS:
        if name in os.environ:
            raise RuntimeError(f"{name} is set")

    root = checkpoint()
    opf = OPF(model=str(root), device="cpu")
    runtime = opf.get_runtime()
    model = runtime.model.float()
    ids = [runtime.encoding.encode(text, allowed_special="all") for text in TEXTS]
    # opf reads a text in windows of n_ctx tokens, runntime whole: every
    # text has to fit in one window. One has to be longer than the tokens a
    # token attends to.
    longest = max(len(text_ids) for text_ids in ids)
    if longest > runtime.n_ctx or longest <= json.loads((root / "config.json").read_text())["sliding_window"]:
        raise RuntimeError(f"the longest text has {longest} tokens")

    spans = []
    for text in TEXTS:
        result = opf.redact(text)
        if result.warning is not None:
            raise RuntimeError(f"{result.warning} {text!r}")
        spans.append(
            [
                {"label": span.label, "start": utf16(text, span.start), "end": utf16(text, span.end)}
                for span in result.detected_spans
            ]
        )

    with torch.inference_mode():
        upstream = [logits(model, text_ids) for text_ids in ids]
        load_hosted(model)
        hosted = [logits(model, text_ids) for text_ids in ids]
        # The same weights in f64 give the reference noise. The experts and
        # the sinks stay in f32: opf computes the experts, the router, the
        # norms, RoPE and the attention softmax in f32 whatever the weights
        # are, and rounds the attention scores to f32 before the softmax. So
        # the noise is f32 against partly f64.
        for name, parameter in model.named_parameters():
            if "mlp1" not in name and "mlp2" not in name and "sinks" not in name:
                parameter.data = parameter.data.double()
        double = [logits(model, text_ids) for text_ids in ids]

    tensors = {}
    for i, text_ids in enumerate(ids):
        tensors[f"ids.{i}"] = np.asarray(text_ids, dtype=np.int32)
        tensors[f"logits.{i}"] = upstream[i]
        tensors[f"logits_hosted.{i}"] = hosted[i]
    pairs = list(zip(hosted, double, strict=True))
    save(
        "privacy-filter",
        {"texts": TEXTS},
        tensors,
        {
            "repo": REPO,
            "revision": REVISION,
            "dtype": "f32",
            "inputs": {"license": "MIT", "count": len(TEXTS)},
            "settings": {"opf": OPF_COMMIT, "hosted": HOSTED_REVISION},
            "noise": {
                "error": max(noise(a, b) for a, b in pairs),
                "cosineDistance": max(cosine_noise(a, b) for a, b in pairs),
            },
        },
        ["torch", "opf", "tiktoken"],
        outputs={"spans": spans},
    )


if __name__ == "__main__":
    main()
