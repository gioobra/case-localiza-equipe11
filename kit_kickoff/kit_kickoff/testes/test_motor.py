"""Testes do motor com os exemplos numéricos do regulamento (6.3.1 a 6.3.6)."""
import sys, unittest
from pathlib import Path
AQUI = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AQUI))
from kickoff import motor, tempo, validar  # noqa: E402
from kickoff.dados import carregar  # noqa: E402
from kickoff.modelo import Contrato  # noqa: E402

DADOS = AQUI / "dados"


def nome(cen, parte):
    return next(m for m in cen.modelos if parte.lower() in m.lower())


@unittest.skipUnless((DADOS / "Bangkok").exists(), "sem dados/Bangkok")
class TestMotor(unittest.TestCase):
    def setUp(self):
        self.cen = carregar(DADOS, "Bangkok", None)
        self.cen.veiculos, self.cen.contratos = [], []

    def motor_com(self, carregadores, politica=None, entrada="b3_45"):
        sub = {"agencia": "Bangkok", "carregadores": carregadores, "entrada": entrada,
               "politica": politica or {}}
        m = motor.Motor(self.cen, validar.normalizar(sub, self.cen))
        m._cs, m._ptr, m._processados, m._pronto = [], 0, set(), True
        m.aberta = True
        return m

    def carro(self, m, vid, parte, soc):
        mod = self.cen.modelos[nome(self.cen, parte)]
        v = motor.Veiculo(vid, mod, mod.rank, soc)
        m.veiculos[vid] = v
        m.patio[vid] = v
        return v

    def test_631_escolha_do_carro(self):
        m = self.motor_com({"ac_7": 1})
        geely = self.carro(m, "a", "Geely", 82)
        mini = self.carro(m, "b", "Mini", 95)
        self.carro(m, "c", "MG4", 100)
        self.carro(m, "d", "JAC", 100)
        m.plugar(mini, m.pontos[0])
        prata = Contrato("k1", "Prata", 2, 600, None, 0, None)
        self.assertEqual(m.escolher_carro(prata, 600).id, "a")
        geely.soc = 78
        self.assertEqual(m.escolher_carro(Contrato("k2", "Prata", 2, 600, None, 0, None), 600).id, "b")
        mini.soc = 70
        self.assertEqual(m.escolher_carro(Contrato("k3", "Prata", 2, 600, None, 0, None), 600).id, "c")

    def test_632_dolphin_gs_dc60(self):
        m = self.motor_com({"dc_60x2": 1}, entrada="se_2000")
        m.limite_kw = None
        v = self.carro(m, "v", "Dolphin GS", 20)
        m.plugar(v, m.pontos[0])
        m.t = 0
        m.recalcular_potencias(0)
        while m.heap:
            t, _, _, tipo, p = __import__("heapq").heappop(m.heap)
            m.avancar(t)
            if tipo == "fim_recarga":
                m.on_fim_recarga(p[0], p[1], t)
        self.assertAlmostEqual(m.t, 36.7, delta=0.06)
        self.assertAlmostEqual(v.soc, 80.0, places=6)

    def test_633_ordem_da_fila(self):
        esperado = {"menor_soc": "ABDC", "chegada": "ABCD", "proxima_retirada": "CDBA"}
        for pr, ordem in esperado.items():
            m = self.motor_com({"ac_7": 1}, {"prioridade": pr})
            m._cs = [Contrato("o", "Ouro", 3, 750, None, 0, None), Contrato("p", "Prata", 2, 795, None, 0, None),
                     Contrato("b", "Bronze", 1, 840, None, 0, None)]
            for vid, parte, soc, t in [("A", "Kwid", 30, 600), ("B", "Mini", 20, 620),
                                       ("C", "MG4", 55, 640), ("D", "Leapmotor", 40, 655)]:
                v = self.carro(m, vid, parte, soc)
                v.na_fila, v.entrou_fila_t = True, t
                m.fila.append(v)
            m.ordenar_fila(660)
            self.assertEqual("".join(v.id for v in m.fila), ordem, pr)

    def test_634_escolha_do_carregador(self):
        m = self.motor_com({"dc_60x2": 1, "ac_22x2": 1})
        g = self.carro(m, "g", "Geely", 30)
        dc = next(p for p in m.pontos if p.id.startswith("dc_60x2"))
        m.plugar(g, dc)
        d = self.carro(m, "d", "Dolphin GS", 20)
        self.assertIs(m.escolher_ponto(d), dc)

    def _cenario_635(self, divisao, limite):
        m = self.motor_com({"dc_60x2": 2}, {"prioridade": "menor_soc", "divisao": divisao})
        m.limite_kw = limite
        c1, c2 = m.pontos
        for vid, parte, soc, p in [("g", "Geely", 30, c1), ("d", "Dolphin GS", 50, c1),
                                   ("k", "Kwid", 40, c2), ("j", "JAC", 75, c2)]:
            m.plugar(self.carro(m, vid, parte, soc), p)
        m.recalcular_potencias(0)
        return {v: round(m.veiculos[v].potencia_kw, 4) for v in "gdkj"}

    def test_635_divisao(self):
        self.assertEqual(self._cenario_635("igual", 40.5), {"g": 10.125, "d": 10.125, "k": 10.125, "j": 10.125})
        self.assertEqual(self._cenario_635("prioridade", 40.5), {"g": 13.9, "d": 0.0, "k": 26.6, "j": 0.0})
        r = self._cenario_635("igual", 80)
        self.assertAlmostEqual(r["j"], 16.45, places=3)
        self.assertAlmostEqual(r["g"], 21.1833, places=3)
        self.assertEqual(self._cenario_635("prioridade", 80), {"g": 53.4, "d": 0.0, "k": 26.6, "j": 0.0})

    def test_636_ponta_zera(self):
        m = self.motor_com({"ac_22x2": 1}, {"evitar_ponta": True})
        v = self.carro(m, "v", "MG4", 29)
        m.plugar(v, m.pontos[0])
        m.em_ponta = True
        m.recalcular_potencias(0)
        self.assertEqual(v.potencia_kw, 0.0)
        self.assertTrue(v.plugado)


if __name__ == "__main__":
    unittest.main()
