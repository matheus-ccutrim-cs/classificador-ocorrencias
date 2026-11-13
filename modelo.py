"""Preparação, treinamento e exportação da MLP."""
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder

SEED = 42
CLASSES = ["Furto", "Roubo"]
FEATURES = ["Bairro", "Período", "Dia da semana", "Tipo de local"]
DAYS = ["SEGUNDA", "TERÇA", "QUARTA", "QUINTA", "SEXTA", "SÁBADO", "DOMINGO"]
CONFIGS = {"A": (32,), "B": (64, 32), "C": (128, 64, 32)}
MISSING = "NÃO INFORMADO"
SOURCE = "https://www.portal.ssp.sp.gov.br/estatistica/consultas"


def normalize(value):
    return re.sub(r"[^A-Z0-9]", "", unicodedata.normalize("NFKD", str(value).upper()))


def prepare(path):
    """Gera um registro por BO. Campos de identificação ficam fora das entradas."""
    path = Path(path)
    if path.suffix.lower() == ".xlsx":
        with pd.ExcelFile(path) as workbook:
            sheets = []
            for name in workbook.sheet_names:
                columns = workbook.parse(name, nrows=0).columns
                if {"CIDADE", "RUBRICA", "NUM_BO"} <= set(columns):
                    sheets.append(workbook.parse(name, usecols=lambda col: col in {
                        "CIDADE", "RUBRICA", "NUM_BO", "ANO_BO", "NOME_DELEGACIA",
                        "ID_DELEGACIA", "DELEGACIA", "DELEGACIA_REGISTRO",
                        "DATA_OCORRENCIA_BO", "DESCR_PERIODO", "BAIRRO", "DESCR_TIPOLOCAL",
                    }))
            if not sheets:
                raise ValueError("Nenhuma aba contém CIDADE, RUBRICA e NUM_BO.")
            raw = pd.concat(sheets, ignore_index=True)
    else:
        raw = pd.read_csv(path, sep=None, engine="python")
    audit = {"Linhas de origem": len(raw)}
    required = ["CIDADE", "RUBRICA", "NUM_BO", "ANO_BO", "DATA_OCORRENCIA_BO",
                "BAIRRO", "DESCR_PERIODO", "DESCR_TIPOLOCAL"]
    missing = set(required) - set(raw)
    if missing:
        raise ValueError(f"Colunas ausentes: {sorted(missing)}")
    raw["data"] = pd.to_datetime(raw["DATA_OCORRENCIA_BO"], format="mixed", dayfirst=True, errors="coerce")
    df = raw.loc[raw["CIDADE"].map(normalize).isin(["SAOPAULO", "SPAULO"]) &
                 raw["data"].dt.year.eq(2024)].copy()
    audit["Linhas na capital em 2024"] = len(df)
    office = next((col for col in ["ID_DELEGACIA", "NOME_DELEGACIA", "DELEGACIA_REGISTRO", "DELEGACIA"]
                   if col in df and df[col].notna().all()), None)
    keys = ["ANO_BO", "NUM_BO"] + ([office] if office else [])
    df = df.dropna(subset=keys).copy()
    audit["Chave de deduplicação"] = " + ".join(keys)
    df["classe"] = df["RUBRICA"].map(normalize).map({"FURTOART155": 0, "ROUBOART157": 1})
    target_counts = df.groupby(keys)["classe"].transform("nunique")
    df = df.loc[df["classe"].notna() & target_counts.eq(1)].copy()
    for name, column in zip([FEATURES[0], FEATURES[1], FEATURES[3]],
                            ["BAIRRO", "DESCR_PERIODO", "DESCR_TIPOLOCAL"]):
        values = df[column].astype("string").str.strip().str.upper().str.replace(r"\s+", " ", regex=True)
        df[name] = values.replace(["", "-", "IGNORADO", "NULL", "NAN"], pd.NA).fillna(MISSING)
    df[FEATURES[2]] = df["data"].dt.dayofweek.map(dict(enumerate(DAYS)))
    # Descarta BOs com descrições de contexto conflitantes e colapsa linhas de itens.
    distinct = df.drop_duplicates(keys + FEATURES)
    clean = distinct.loc[~distinct.duplicated(keys, keep=False), FEATURES + ["classe"]].copy()
    clean["classe"] = clean["classe"].astype(int)
    audit["BOs elegíveis"] = len(clean)
    if len(clean) < 100 or clean["classe"].value_counts().reindex([0, 1], fill_value=0).min() < 20:
        raise ValueError("Base insuficiente após o recorte. Confira período, cidade e rubricas.")
    if len(clean) > 15000:
        clean, _ = train_test_split(clean, train_size=15000, stratify=clean["classe"], random_state=SEED)
    audit["BOs na amostra"] = len(clean)
    return clean, audit


def split(data):
    train, rest = train_test_split(data, test_size=.30, stratify=data["classe"], random_state=SEED)
    val, test = train_test_split(rest, test_size=.50, stratify=rest["classe"], random_state=SEED)
    encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=10, dtype=np.float32)
    encoder.fit(train[FEATURES])
    return train, val, test, encoder


def metrics(y, predictions):
    p, r, f, _ = precision_recall_fscore_support(y, predictions, average="macro", zero_division=0)
    return dict(acuracia=accuracy_score(y, predictions), precision=p, recall=r, f1=f)


def experiment(train, val, test, encoder):
    x_train, x_val = (encoder.transform(part[FEATURES]) for part in (train, val))
    y_train, y_val = train["classe"].to_numpy(), val["classe"].to_numpy()
    runs, models = {}, {}
    for name, layers in CONFIGS.items():
        mlp = MLPClassifier(hidden_layer_sizes=layers, activation="relu", solver="adam",
                            alpha=.001, learning_rate_init=.001, batch_size=128, random_state=SEED)
        history = []
        for epoch in range(1, 31):
            mlp.partial_fit(x_train, y_train, classes=np.array([0, 1]))
            train_prob, val_prob = mlp.predict_proba(x_train), mlp.predict_proba(x_val)
            history.append(dict(epoca=epoch, treino_loss=log_loss(y_train, train_prob),
                                val_loss=log_loss(y_val, val_prob),
                                treino_acc=accuracy_score(y_train, train_prob.argmax(axis=1)),
                                val_acc=accuracy_score(y_val, val_prob.argmax(axis=1))))
        models[name] = mlp
        runs[name] = dict(camadas=layers, historico=history, **metrics(y_val, mlp.predict(x_val)))
    winner = max(runs, key=lambda name: (runs[name]["f1"], runs[name]["acuracia"], -len(CONFIGS[name])))
    model = models[winner]
    y_test = test["classe"].to_numpy()
    pred = model.predict(encoder.transform(test[FEATURES]))
    majority = int(np.bincount(y_train).argmax())
    result = dict(arquiteturas=runs, selecionada=winner, teste=metrics(y_test, pred),
                  baseline_validacao=metrics(y_val, np.full(len(y_val), majority)),
                  matriz=confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
                  divisao={name: len(part) for name, part in zip(["Treino", "Validação", "Teste"], [train, val, test])},
                  classes={name: int(sum(part["classe"].eq(i).sum() for part in [train, val, test]))
                           for i, name in enumerate(CLASSES)})
    return model, result


def export_web(encoder, model):
    """Exporta índices one-hot e parâmetros das camadas densas para JavaScript."""
    maps = []
    for i, categories in enumerate(encoder.categories_):
        infrequent = encoder.infrequent_categories_[i]
        frequent = [str(x) for x in categories if infrequent is None or x not in infrequent]
        mapping = {category: len(maps) + index for index, category in enumerate(frequent)}
        if infrequent is not None:
            mapping.update({str(x): len(maps) + len(frequent) for x in infrequent})
        maps.extend([mapping] + [None] * (len(frequent) + int(infrequent is not None) - 1))
    mappings = [mapping for mapping in maps if mapping is not None]
    for sample in zip(*(list(mapping)[:5] for mapping in mappings)):
        inputs = pd.DataFrame([sample], columns=FEATURES)
        vector = np.zeros(model.n_features_in_)
        vector[[mapping[value] for mapping, value in zip(mappings, sample)]] = 1
        np.testing.assert_array_equal(vector, encoder.transform(inputs).toarray()[0])
    return dict(features=FEATURES, classes=CLASSES, mappings=mappings,
                size=model.n_features_in_, weights=[w.tolist() for w in model.coefs_],
                biases=[b.tolist() for b in model.intercepts_])


def save(path, out, demo=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data, audit = prepare(path)
    train, val, test, encoder = split(data)
    model, result = experiment(train, val, test, encoder)
    result.update(auditoria=audit, demo=demo)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "resultados.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    payload = export_web(encoder, model)
    payload["demo"] = demo
    (out / "modelo.js").write_text("window.MODELO = " + json.dumps(payload, ensure_ascii=False) + ";", encoding="utf-8")
    history = pd.DataFrame(result["arquiteturas"][result["selecionada"]]["historico"])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout="constrained")
    for ax, metric, title in zip(axes, ["acc", "loss"], ["Acurácia", "Entropia cruzada"]):
        for subset, label in [("treino", "Treino"), ("val", "Validação")]:
            ax.plot(history["epoca"], history[f"{subset}_{metric}"], label=label)
        ax.set(xlabel="Época", title=title)
        ax.legend()
    matrix = np.array(result["matriz"])
    axes[2].imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        axes[2].text(j, i, str(value), ha="center", va="center",
                     color="white" if value > matrix.max() / 2 else "black")
    axes[2].set(xticks=[0, 1], yticks=[0, 1], xticklabels=CLASSES, yticklabels=CLASSES,
                xlabel="Estimado", ylabel="Real", title="Matriz de confusão · teste")
    fig.savefig(out / "avaliacao.png", dpi=170)
    plt.close(fig)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dados")
    parser.add_argument("--saida", default="docs")
    args = parser.parse_args()
    results = save(args.dados, args.saida)
    print(json.dumps({key: results[key] for key in ["auditoria", "selecionada", "teste"]}, ensure_ascii=False))
