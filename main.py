import cv2
import matplotlib.pyplot as plt
from utils import filtros, morfologia, segmentacao, medidas_objetos, rastreamento_video

def menu():
    img1 = cv2.imread('root.jpg')
    img2 = cv2.imread('linhas.png')
    img3 = cv2.imread('arara.png')
    img4 = cv2.imread('ruidos.png')
    img5 = cv2.imread('unequalized.jpg')

    if img1 is None or img2 is None:
        print("Erro ao carregar imagens")
        return

    while True:
        print("\n--- Menu de Operações PDI ---")
        print("1. Conversões em img1 (root.png)")
        print("2. Filtros e bordas em img1 (root.png)")
        print("3. Morfologia em im4g4 (ruidos.png)")
        print("4. Análise em img5 (unequalized.jpg)")
        print("5. Rastreamento de objeto (câmera)")
        print("0. Sair")
        escolha = input("Escolha: ")

        if escolha == "0":
            break

        elif escolha == "1":
            print("1. Tons de cinza")
            print("2. Negativo")
            print("3. Binarização (Otsu)")
            op = input("Escolha: ")
            if op == "1":
                cinza = cv2.imread(img3, cv2.COLOR_GRAYSCALE)
                cv2.imshow("Cinza - img3", cinza)
            elif op == "2":
                negativo = 255 - img3
                cv2.imshow("Negativo - img3", negativo)
            elif op == "3":
                cinza = cv2.imread(img3, cv2.COLOR_GRAYSCALE)
                binaria = segmentacao.otsu_binarizacao(cinza)
                cv2.imshow("Binária Otsu - img3", binaria)
            else:
                print("Opção inválida.")
            cv2.waitKey(0)

        elif escolha == "2":
            print("1. Média")
            print("2. Mediana")
            print("3. Canny")
            op = input("Escolha: ")
            cinza = cv2.imread(img1, cv2.COLOR_GRAYSCALE)
            if op == "1":
                suavizada = filtros.media(cinza)
                cv2.imshow("Média - img1", suavizada)
            elif op == "2":
                suavizada = filtros.mediana(cinza)
                cv2.imshow("Mediana - img1", suavizada)
            elif op == "3":
                bordas = filtros.canny(cinza)
                cv2.imshow("Canny - img1", bordas)
            else:
                print("Opção inválida.")
            cv2.waitKey(0)

        elif escolha == "3":
            print("Operações morfológicas:")
            tipo_imagem = input("Escolha o tipo de imagem: 1 - binária, 2 - cinza: ")

            while tipo_imagem not in ["1", "2"]:
                print("Opção inválida. Escolha 1 ou 2.")
                tipo_imagem = input("Escolha: ")

            if tipo_imagem == "1":
                cinza = segmentacao.otsu_binarizacao(img4)
            else:
                cinza = cv2.imread(img3, cv2.COLOR_GRAYSCALE)

            print("1. Erosão")
            print("2. Dilatação")
            print("3. Abertura")
            print("4. Fechamento")
            op = input("Escolha: ")
            if op == "1":
                resultado = morfologia.erosao(cinza)
                cv2.destroyAllWindows()
            elif op == "2":
                resultado = morfologia.dilatacao(cinza)
                cv2.destroyAllWindows()
            elif op == "3":
                resultado = morfologia.abertura(cinza)
                cv2.destroyAllWindows()
            elif op == "4":
                resultado = morfologia.fechamento(cinza)
                cv2.destroyAllWindows()
            else:
                print("Opção inválida.")
                continue
            cv2.imshow("Resultado Morfologia - img2", resultado)
            cv2.waitKey(0)

        elif escolha == "4":
            cinza = cv2.imread(img5, cv2.IMREAD_GRAYSCALE)
            binaria = segmentacao.otsu_binarizacao(cinza)
            rotulada, num = segmentacao.crescimento_regiao(binaria)
            medidas = medidas_objetos.medidas_por_objeto(rotulada)
            histograma = medidas_objetos.histograma(rotulada)
            print("\n--- Medidas dos Objetos ---")

            medidas_objetos.plot_histogram(histograma, "Histograma de linhas.png", "histograma_unequalized")

            for m in medidas:
                print(f"Objeto {m['rotulo']}: Área={m['area']}, Perímetro={m['perimetro']:.2f}, Diâmetro={m['diametro']:.2f}")
            print(f"Total de objetos: {num}")

        elif escolha == "5":
            rastreamento_video.rastrear_video()

        else:
            print("Opção inválida.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    menu()
