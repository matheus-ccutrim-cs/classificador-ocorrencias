"""Gera um PDF de dez slides com resultados reais, quando disponíveis."""
import argparse
import json
from pathlib import Path

from reportlab.lib.colors import HexColor, white
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

from modelo import CONFIGS, SOURCE


def build(out, results, screenshot, members):
    path = Path(results)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if data and data.get("demo"):
        raise ValueError("Resultados sintéticos não podem compor o relatório oficial.")
    canvas = Canvas(str(out), pagesize=(960, 540))
    canvas.setTitle("Ocorrências SP | Global Solution")
    number = 0

    def slide(title):
        nonlocal number
        number += 1
        canvas.setFillColor(HexColor("#F0F5F8"))
        canvas.rect(0, 0, 960, 540, fill=1, stroke=0)
        canvas.setFillColor(HexColor("#102B41"))
        canvas.rect(0, 397, 960, 143, fill=1, stroke=0)
        canvas.setFillColor(HexColor("#83E1CC"))
        canvas.setFont("Helvetica-Bold", 13)
        canvas.drawString(54, 498, "GLOBAL SOLUTION  /  APRENDIZADO PROFUNDO")
        canvas.setFillColor(white)
        canvas.setFont("Helvetica-Bold", 32)
        canvas.drawString(54, 441, title)
        canvas.setFillColor(HexColor("#496477"))
        canvas.setFont("Helvetica", 11)
        canvas.drawString(54, 30, "FIAP  •  Segurança pública inteligente em São Paulo")
        canvas.drawRightString(905, 30, f"{number:02d} / 10")

    def lines(*items, y=355, step=42):
        canvas.setFillColor(HexColor("#142B3D"))
        canvas.setFont("Helvetica", 17)
        for item in items:
            canvas.drawString(55, y, item)
            y -= step

    def image(path, x, y, width, height):
        canvas.drawImage(ImageReader(str(path)), x, y, width, height,
                         preserveAspectRatio=True, anchor="c", mask="auto")

    slide("Roubo ou furto de celular?")
    lines("Classificação de ocorrências registradas com redes densas.",
          "Recorte: cidade de São Paulo, ocorrências de 2024.",
          "Objetivo: comparar três MLPs e integrar a selecionada a um formulário web.",
          f"Integrantes: {members}")
    canvas.showPage()

    slide("Base pública e recorte")
    lines("SSP-SP · Celulares subtraídos · edição 2024",
          "CIDADE e data do fato delimitam o recorte geográfico e temporal.",
          "RUBRICA define o alvo: art. 155 → Furto | art. 157 → Roubo.",
          "Bairro e tipo de local descrevem o contexto espacial.",
          "Período e dia da semana descrevem o contexto temporal.",
          f"BOs na amostra: {data['auditoria']['BOs na amostra']:,}" if data else
          "Amostra planejada: até 15.000 BOs, com estratificação por classe.")
    lines(SOURCE, y=91)
    canvas.linkURL(SOURCE, (55, 88, 520, 111), relative=0)
    canvas.showPage()

    slide("Preparação dos dados")
    lines("Agrupamento por chave do BO para consolidar múltiplos itens.",
          "Exclusão de conflitos de rótulo e contexto e de datas inválidas.",
          "Categorias ausentes recebem NÃO INFORMADO. Campos pessoais ficam fora.",
          "Treino 70%, validação 15% e teste 15%, com estratificação e semente 42.",
          "One-hot ajustado no treino. Categorias com menos de 10 casos são agrupadas.",
          "As entradas binárias já estão em 0/1 e dispensam escalonamento.")
    if data:
        lines("  |  ".join(f"{name}: {n:,}" for name, n in data["divisao"].items()), y=94)
    canvas.showPage()

    slide("Como a MLP classifica")
    lines("Entrada: vetor one-hot com uma posição por categoria codificada.",
          "Cada camada densa calcula uma soma ponderada e adiciona um viés.",
          "Camadas ocultas: ReLU(z) = max(0, z) para aprender relações não lineares.",
          "Saída: um neurônio sigmoide fornece a probabilidade estimada de Roubo.",
          "A probabilidade de Furto é o complemento. O limiar de decisão é 0,5.")
    canvas.showPage()

    slide("Configuração do treinamento")
    lines("Entropia cruzada binária penaliza previsões erradas e excessivamente confiantes.",
          "Adam usa momentos dos gradientes para ajustar a atualização dos pesos.",
          "Taxa de aprendizado inicial 0,001  |  L2 = 0,001",
          "30 épocas  |  minilotes de 128 exemplos  |  semente 42",
          "Mesma divisão, codificação e configuração para as três arquiteturas.")
    canvas.showPage()

    slide("Três arquiteturas comparadas")
    lines("A  /  32 neurônios: referência de menor capacidade.",
          "B  /  64 → 32 neurônios: testa o ganho de uma camada adicional.",
          "C  /  128 → 64 → 32 neurônios: testa uma rede mais profunda e larga.",
          "A capacidade aumenta de A para C. A validação mede o efeito dessa mudança.",
          "Referência adicional: prever sempre a classe majoritária do treino.")
    canvas.showPage()

    slide("Desempenho na validação")
    headings = ["Modelo", "Acurácia", "Precision", "Recall", "F1"]
    xs = [55, 270, 430, 590, 750]
    for x, name in zip(xs, headings):
        canvas.setFont("Helvetica-Bold", 16)
        canvas.drawString(x, 354, name)
    for i, name in enumerate(CONFIGS):
        row = data["arquiteturas"][name] if data else None
        values = [name] + [f"{row[key]:.3f}" if row else "Pendente" for key in
                           ["acuracia", "precision", "recall", "f1"]]
        for x, value in zip(xs, values):
            canvas.setFont("Helvetica", 16)
            canvas.drawString(x, 311 - 39 * i, value)
    lines("Precision, recall e F1 usam média macro entre Furto e Roubo.",
          "Seleção por F1 macro. Desempates por acurácia e menor profundidade.",
          (f"Selecionada: {data['selecionada']}  |  F1 majoritário na validação: "
           f"{data['baseline_validacao']['f1']:.3f}") if data else
          "Execute o treinamento com a planilha oficial para preencher esta tabela.",
          y=165, step=34)
    canvas.showPage()

    slide("Curvas e avaliação final")
    if data:
        image(path.with_name("avaliacao.png"), 45, 151, 870, 230)
        selected = data["arquiteturas"][data["selecionada"]]["historico"]
        last = selected[-1]
        gap = last["treino_acc"] - last["val_acc"]
        increase = last["val_loss"] - min(row["val_loss"] for row in selected)
        diagnosis = ("A perda de validação voltou a subir. Compare com a curva de treino."
                     if increase > .02 else "A perda de validação terminou próxima do seu mínimo.")
        lines(f"Teste: acurácia {data['teste']['acuracia']:.3f}  |  F1 macro {data['teste']['f1']:.3f}",
              f"Na última época, diferença treino − validação: {gap:+.1%} na acurácia.",
              diagnosis, y=136, step=29)
    else:
        lines("Curvas e matriz de confusão serão inseridas após o treinamento oficial.",
              "Treino melhora e validação piora: possível sobreajuste.",
              "Treino e validação fracos: investigar capacidade e sinal dos atributos.",
              "Matriz no teste: linhas indicam classes reais, colunas indicam estimativas.")
    canvas.showPage()

    slide("Interface web e inferência")
    if screenshot and Path(screenshot).exists():
        image(screenshot, 49, 63, 430, 320)
        canvas.setFillColor(HexColor("#142B3D"))
        canvas.setFont("Helvetica", 15)
        for y, value in [(325, "Quatro seletores reproduzem as entradas da MLP."),
                         (283, "A página aplica one-hot, ReLU e sigmoide."),
                         (241, "A classificação exibe as duas probabilidades."),
                         (199, "A inferência roda no navegador.")]:
            canvas.drawString(505, y, value)
    else:
        lines("O formulário usa as categorias observadas no treinamento.",
              "O navegador carrega a codificação, os pesos e os vieses da MLP.",
              "A interface mostra a classe estimada e as duas probabilidades.",
              "Adicione uma captura da aplicação usando a opção --captura.")
    canvas.showPage()

    slide("Conclusões e limites")
    lines("A proposta classifica o tipo de um registro de celular subtraído.",
          "Os dados representam ocorrências registradas e estão sujeitos à subnotificação.",
          "O recorte de 2024 limita a extrapolação a outros anos, cidades e objetos.",
          "As probabilidades expressam a saída do modelo, sem calibração adicional.",
          "IA generativa apoiou código, documentação e validação técnica do fluxo.",
          "A interpretação empírica depende da execução sobre os dados oficiais.")
    canvas.showPage()
    canvas.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--saida", default="Relatorio_GS.pdf")
    parser.add_argument("--resultados", default="docs/resultados.json")
    parser.add_argument("--captura", default=None)
    parser.add_argument("--autor", default="[Matheus Costa Cutrim 568087]")
    args = parser.parse_args()
    build(args.saida, args.resultados, args.captura, args.autor)
