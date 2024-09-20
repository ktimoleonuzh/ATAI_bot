import rdflib
from src.utils import load_data_config

data_config = load_data_config()

# For SPARQL queries
header = '''
    prefix wdt: <http://www.wikidata.org/prop/direct/>
    prefix wd: <http://www.wikidata.org/entity/>
    prefix schema: <http://schema.org/> 
    prefix ddis: <http://ddis.ch/atai/>'''

namespace_map = {
        'wd': rdflib.Namespace(data_config['namespaces']['WD']),
        'wdt': rdflib.Namespace(data_config['namespaces']['WDT']),
        'schema': rdflib.Namespace(data_config['namespaces']['SCHEMA']),
        'rdfs': rdflib.Namespace(data_config['namespaces']['RDFS']),
        'ddis': rdflib.Namespace(data_config['namespaces']['DDIS'])
    }

WD = namespace_map['wd']
WDT = namespace_map['wdt']
SCHEMA = namespace_map['schema']
RDFS = namespace_map['rdfs']
DDIS = namespace_map['ddis']

# MOVIE ENTITY DEFINITIONS
film_entities = {
    'animated feature film': 'Q29168811',
    'animated film': 'Q202866',
    'film': 'Q11424',
    '3D film': 'Q229390',
    'live-action/animated film': 'Q25110269'
}

special_chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ': ', ':', '!', '-']

