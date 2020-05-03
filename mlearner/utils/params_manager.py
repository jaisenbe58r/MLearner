
"""Jaime Sendra Berenguer-2018-2022.
MLearner Machine Learning Library Extensions
Author:Jaime Sendra Berenguer<www.linkedin.com/in/jaisenbe>
License: MIT
"""

import json


class ParamsManager(object):
    """
    Objeto utilizado para leer los datos de un fichero json de configuracion.

    Parameters
    ----------

        - params_file: fichero .json desde el directorio de trabajo.
        - key_read: argumento "key" del fichero .json sobre el que queremos actuar.

    Este objeto nos proporciona los siguientes métodos:

        - get_params(): obtener parametros a partir del "key" especificado
        - update_params(): actualizacion de los valores de un "key" especificado.
        - export_params(): Escribir parametros en un archivo
    """

    def __init__(self, params_file, key_read="info_json"):
        self.params = json.load(open(params_file, "r"))
        self.key_read = key_read

    def get_params(self):
        return self.params[self.key_read]

    def update_params(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.get_params().keys():
                self.params[self.key_read][key] = value

    def export_params(self, filename):
        with open(filename, "w") as f:
            json.dump(
                self.params[self.key_read],
                f,
                ident=4,
                separators=(",", ":"),
                sort_keys=True,
            )
            f.write("\n")
