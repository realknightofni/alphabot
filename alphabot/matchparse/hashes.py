import cv2
import imagehash

from collections import namedtuple, defaultdict
from functools import lru_cache, partial

from PIL import Image

from .base import DEFAULT_HASH_SIZE

Artifact = namedtuple('Artifact', ['name', 'reference_hash'])
Hero = namedtuple('Hero', ['name', 'reference_hash'])
Trait = namedtuple('Trait', ['name', 'hero_name', 'reference_hash', 'color'])
Genre = namedtuple('Genre', ['name', 'reference_hash'])

UNKNOWN_ARTIFACT = Artifact('UNKNOWN_ARTIFACT', '_')
UNKNOWN_HERO = Hero('UNKNOWN_HERO', '_')
UNKNOWN_TRAIT = Trait('UNKNOWN_TRAIT', 'UNKNOWN_HERO', '_', 'UNKNOWN_COLOR')
UNKNOWN_GENRE = Genre('UNKNOWN_GENRE', '_')

def is_unknown(obj):
    if obj is None:
        return True
    elif isinstance(obj, Artifact):
        return obj == UNKNOWN_ARTIFACT
    elif isinstance(obj, Hero):
        return obj == UNKNOWN_HERO
    elif isinstance(obj, Trait):
        return obj == UNKNOWN_TRAIT
    elif isinstance(obj, Genre):
        return obj == UNKNOWN_GENRE
    else:
        raise ValueError(f'Unknown object type: {type(obj)}')

def is_known(obj):
    return not is_unknown(obj)


def count_known_vs_total(guesses):
    total = len(guesses)
    known = sum(is_known(guess) for guess in guesses)

    return known, total


HAMMING_TRAIT_THRESHOLD = 25
HAMMING_HERO_THRESHOLD = 10
HAMMING_ARTIFACT_THRESHOLD = 15
HAMMING_GENRE_THRESHOLD = 25


TRAITS = {
    "Alicia": {
        "yellow": {
            "Hypothermia": ["0606638db95d57b1f8741e060"],
            "Bone Chill": ["040442c97f5e5791c1fe7f800"],
        },
        "green": {
            "Snow Maker": ["000407e1fef67e97c9e233030", "010203e1fe7a1ed7e4f219010"],
            "Accelerated Freezing": ["020581e05f7d5fd4dca211010"],
            "Cold Crystal": ["000003f8ee7fdff6fc9211010"],
        },
        "blue": {
            "Winter Orb": ["0405c3607e4bcfd3f8361d010", "080b86c0fc979fa7f06c18020"],
            "Ice Soul Sword": [
                "0c0cc709f29e6f94f11e31030",
                "040c4389f95e57d47c8e11810",
            ],
        },
        "red": {
            "Fitzgerald": ["080fc4c878bf6fd3f1722f020", "040762e43d5f97e1f0b915010"]
        },
    },
    "Alloya": {
        "yellow": {
            "Drone Detonation": ["080984724eb1af87e0fc26000"],
            "Raging Inferno": ["0407c249e35cd3b6fc0f0383c"],
        },
        "blue": {
            "Battle Order": ["080885d2fa9cbfedd8702a020", "0406636c7d4e1fb2ecbc17018"],
            "Part Reuse": ["0407c248107fdff0409217010"],
            "Internal Maintenance": ["0c0ec4da3a9ae67f698633030"],
            "Endless Fuel Belt": ["0c0fc41b328ce73dc9763f030"],
        },
        "red": {"Infinity Command": ["080b05d0f8efbfe3e1743f020"]},
        "green": {"Protective Mech": ["110d63f8fe71dff3f8c611000", "221ac7f1fce3bfe7f18c22000"]},
    },
    "Asuka": {
        "green": {"Pheromone": ["000000e0fe7fdff6648319818"]},
        "yellow": {
            "Ninth Tail": ["080884127abfbfe6719c26000", "0404420c3d5f9ff33cc613000"],
            "Onibidama": ["0606626dc344dd82687e1f018"],
            "Sessho-Seki": [
                "04044389ed77cff3f87c1f018",
                "0c08c71bbeff9fe3f0f81e010",
                "08088711deef9fe3f0f83e030",
            ],
            "Night Hunt": ["0c088653d6ffaaa3c0781e040"],
        },
        "blue": {"Taizoukai": ["040743e0660fcb73d8f418010"],
                 "Pluviophile": ["0c08c42842806ff7f19e21030"],},
    },
    "Brynhild": {
        "green": {
            "Layered Peaks": ["080703f1fcbeada7f9c422020", "040381f8fe3f16d7fce611010"],
            "Hill Mover": ["100601d8d6e1a367f9fe23030", "080300ec6b70d1b3f4ff11018"],
        },
        "blue": {
            "Counterstrike": [
                "0405c238b75dd776dc861f010",
                "0404c238375dd776dc8617010",
                "080b84716ebbaeedb90c3e020",
                "0c09c431eebba6e5b91c3e000",
            ]
        },
        "yellow": {
            "Sunset Flare": ["040443292b7f5fd1f8f80d000", "0c08c65a56ffffa3f8b81a000"],
            "Eilifvorn's Sunrise": [
                "0c0c46e97fff7fd3f00000000",
                "1e08c5e9fefffff3f07800000",
            ],
        },
        "red": {
            "High Mountain": ["0c0fc619ce7ed79609863f030", "0607e20cc77e53d20cc31f818", "e030084246b66398023ac0381"],
            "Evergreen": ["080f841272aebeaff97c3e020", "0407c2093a565ed7f8be1f010"],
            "Final Faith": ["0405c2697d46511444ba1f010"],
        },
    },
    "Cull": {
        "green": {
            "Countdown": ["1f0041e038ce7494cdbe3f030", "1f0441f0384e525664ff1f818"],
            "Deathcall": ["1c0f83f1b4ede795e97233030", "0e07c1f8da7693d0e4bb11010"],
            "Cutthroat": ["301d060390e67db789fe3f020", "180c0709c0735e9784fa1f010"],
        },
        "yellow": {
            "Reap": ["0607e30cff72d440bc2f05810", "0c0fc619fff4e451383e0f814"],
            "Heartbreak": ["08088771febf2f81c000770f8"],
            "Exsanguinate": [
                "080fc7f9fedea4a0200802000",
                "080fc7f9fedee4b0200802000",
                "0407e3fcff6f5250100401000",
            ],
        },
        "blue": {
            "Faceless": ["0405c2390d5f57d4348e17010"],
            "Candlelight": ["0c0fc409228c6793e97e3b030", "04074209314c5795f4fe19010"],
        },
    },
    "Emrald": {
        "blue": {
            "Amber": ["080f83f1fcffbfe7f1fc3e000"],
            "Jade": ["0c0ac498329eefbbe9763f030"],
        },
        "green": {"Obsidian": ["120cc7f1feede794f93223030"],
                  "Crystal": [],},
        "yellow": {"Amethyst": ["0c0cc49b7a9ee793e0780e000"]},
        "red": {
            "Ruby": ["080b87e8fcbf6ec371f82c020", "0407c3f87e3fce61b0fc15010"],
            "Topaz": ["080984d372dca7a9f12c2a020", "0404e27cbd7e53c0fc9315810"],
            "Sapphire": ["000007c0f83e0dc0f03c06020"],
        },
    },
    "Hellsing": {
        "green": {"Evil Spreads": ["00186738ccb36794c90233030"]},
        "blue": {
            "Colin's Apple": ["0406c260b95f97e2f4b819010"],
            "Etherealization": ["080c8331fedcb22c09323a030"],
            "Fragility": ["00000380e87b1ec7f0f83e000", "000c0180ec3d0f47f8fc3f000"],
            "Justice Must Fail": [
                "08088411543e2fa3e07022020",
                "0c08c419debe6fd1e93223030",
            ],
        },
        "yellow": {
            "Massacre": ["0c0885d97eff2f81e0781c030"],
            "Dead Scream": ["0c0cc7f9cec0e796d8fc1e030", "040442f9c760d7d358fe0e038"],
        },
        "red": {"Devour Fear": ["0c0fc5e878806316d1ce3f030"]},
    },
    "Jacquelyn": {
        "blue": {
            "Noob Mark": ["040540e57d7fdff5f03815010"],
            "Shadow Cutter": ["0404424db96e5396ec921f010"],
            "Shadow Dancer": ["080e84f2fa8ca329e95436020"],
            "Dynamic Vision": ["0c0781e078be0781e03000000"],
        },
        "yellow": {"Afterimage Blade": ["0606634db14855c20c0f3f8e0"]},
        "green": {
            "Instinctive Perception": ["150fc3b8c65b5294e5b21b010", "3f0cc732cb96d324d8ec0c000"],
            "Home Game": ["000381f06c5b56d5f48211010"],
            "Rift Cloak": ["000601b06e3f5dd6659a31010", "180481f2faf6dba4f8cc0c000"],
        },
    },
    "James": {
        "blue": {
            "Piercing Bullet": [
                "0407632cbd57dff07cf91f018",
                "080e84937eafbfe8f9f43e020",
            ]
        },
        "red": {
            "Walking Target": ["040dc3e87e030703e0d833010"],
            "Ruthless": ["04044209bb7fd7d5b4fe35810", "0808c41b36ffa7a969fc3f020"],
            "Perfect Execution": [
                "040cc5a9305a5cf318ce33030",
                "080dc5c872b6bce631cc36020",
                "80400040380e0781f87f07eff",
            ],
        },
        "yellow": {
            "Decisive Eradication": ["0407c3fcc766d993fcfe1f818"],
            "Skilled Reload": ["08088411aeebbac7f1bc6b000"],
        },
        "green": {
            "Chain Plan": ["0c0707f1acaf3ce7f93422020", "040383f8d6359ef7fcba11010", "0c0fc738bceb9ce5d83432000"],
            "Bullet Rain": ["0404c098b65758f71cc211010", "040485b1b6b654f618c213030", "040d85b1b6b6f0e718c41e000"],
        },
    },
    "Jenny": {
        "yellow": {
            "Sunbathing": ["0606c20d395f5ef1f03804000"],
            "Lay on Hands": ["0c0cc41b0affe7b1e03c00000"],
        },
        "blue": {
            "Divine Affinity": [
                "0407c3f9fe7fdff7f8fe1f010",
                "080f87f1feffbfe7f9fc3e020",
                "1e0fc7f9feff9fe7f8fc0c000",
            ],
            "Fast Healing": ["08088771fc7718c3607008020", "040443b8fe3b8c61b03804010"],
            "Rain of Light": ["080a803378fcaf23b9241a020", "0405400cbd7e13c1dcd31f010"],
        },
        "green": {"Enhanced Favor": ["00182619fe7ffcf6190233030"],
                  "Holy Light Surge": ["0003c1b8da7edd77bcfb11018", "00078371b4fdbaff79f623030"],},
        "red": {"Aegis of God": ["040103f8ee5f56d1b4ba15010"]},
    },
    "Kay": {
        "red": {
            "Mech:Counteract": [
                "080f84f11edbb9e7f9fc3e020",
                "000180700c218ce1f03c03110",
            ],
            "Mech:Resist": ["080f87f19caba76be9743e020", "0407c3fcc756d190f4bf1f810"],
        },
        "yellow": {
            "Mech:Demolish": ["04044209c97fde71b83e0f000", "0808841392ffbce3707c1e000"],
            "Ultra-Nanotech": ["0c0fc51b7adefff5e97816020"],
        },
        "green": {"Mech:TurtleShell": ["000783f1fcffbfe7f9fc3e020"]},
        "blue": {
            "Mech:Protect": ["0007c3f8ff7fdff3fcfe1f010", "000f83f9fe7f9fe7f8fc3e000"],
            "Mech:Tenacious": [
                "080001c0f8ffafa1c0702a020",
                "040540e87d7fc7d0e03805010",
            ],
            "Mech:Hardening": ["040442090143d5f47c8e13010"],
        },
    },
    "Malachite": {
        "red": {
            "Topological Shooting": [
                "0c0cc6090df27cd4398e31830",
                "00000000f63ecdf67dfe2f1fc",
                "1e0844090eb4ecb4790e33030",
            ],
            "Fully Armed": ["08088412269c9fc3e0701c020", "04044209114e4fe1f0380e010", "0c000409328c8fc3e0781c020"],
            "Vent": ["04040318bf3bc4d0f48c1f010"],
            "Valor Medal": ["080f87f1fcbea22889742a020", "0407e3f8ff4f1100449b17810"],
            "Behemoth": ["0407c3f83964d835b4ee11010", "080fc7f032e9b86329cc33020"],
            "Annihilation Conviction": ["0c0847e8fabee723f9fc3e020"],
        },
        "green": {"Gigantification": ["0103c1e0f06f13768ce319818"]},
        "yellow": {
            "Seismic Aftershock": [
                "040442094952d4718ce73f87c",
                "0808841292a5a8e31bce7f0f8",
            ]
        },
    },
    "Merlina": {
        "yellow": {
            "Spark Rhythm": ["040442099371def3b8c604000", "1e0844dbaef3dde330300c000"],
            "Plasma Blast": ["0c0dcc1b82bceff1b8320d01c"],
        },
        "blue": {
            "Deep Shock": ["080c85d2fabea4a359dc7f020"],
            "Crackling Lance": ["040642c8d95e53b478be1b810"],
            "Galvanic Field": ["0607624d814b57d6a4831b818", "3e1a44ca029ef7a4188c3e000"],
            "Arc Magic": ["060ec6c9194350741d8e17030"],
        },
        "green": {"Thunderstorm": ["000063f8384751d63c8319818"],},
        "red": {"Lightning Rod": ["0c0c47f91c87f39581ca37030"]},
    },
    "Mina": {
        "green": {
            "Hyper-Cryo Circuit": [
                "000201687f6fddf7bcef1d818",
                "000002d0fedf7bf779de3b030",
            ]
        },
        "yellow": {
            "Icy Snowball": ["08088413e2aaada530dc1f038", "1e0846c9bab6dcc2307c0f010"],
            "Noble Snowman": ["08088712e29cb723f0f83e070"],
            "Snow Fairy": ["04044209f37f5791e0780c000", "08088413e6feaf23c0f018000"],
            "Ski Resort": ["0e0fe60d6f4f53d3603802004", "3f18451b7ecef3a2403002000", "0c0dc41b5e8ee3b6c0700600c"],
        },
        "blue": {
            "Winter Sonata": ["0404000000000301b82e01010", "060662cc2d415311fcee11018"],
            "Bubble Serenade": [
                "040c034036060dd3744011010",
                "0404436cbf4755d374ea19810",
                "080806807c8c0ba6e88032020",
                "1e0807f87e86dfa6e8c41e000",
            ],
        },
        "red": {"Glacial Aria": ["0604636cc57216331cb31d818", "0808c6d98ae4ec7639663b030"]},
    },
    "Moriatee": {
        "yellow": {"Poison Lingering": ["08088671fe9fbfe3f0701e008"],
                   "Terminal Illness": ["040543c9f95e57c1fcf01c030"]},
        "blue": {
            "Aggregated Outbreak": ["040442096d5f5114e4ba1b010"],
            "Accelerated Onset": [
                "080f87f8b4be6790c9323e030",
                "0c0fc7f8b49e6790c1323f030",
            ],
            "Toxic Hormone": ["060463187d4e5391f4c711818"],
        },
        "green": {
            "Inflammatory": ["0081f0783c5c5f1704c31b818"],
            "Black Serpent": ["1e0dc1a078e3b66e397c22020", "0f06c1a03cf3fa77397e23030", "0703e0d01e71dd339cbe11010"],
        },
        "red": {"Res-Pathogen": ["0406c3b93d465314e5ef1b010",
                                 "080d8f725a84a429cbdc36020",]},
    },
    "Samuel": {
        "blue": {
            "Praetor": ["080c8713c8f6bf2fc1f836020"],
            "Combative Instinct": [
                "08088413c4f33f8fe1fc2e020",
                "04044308e37c9fe3f0fe13010",
            ],
        },
        "green": {
            "Herald of Rage": ["380700c0e03a57d5fcba11010"],
            "Beast Fangs": ["000ff36cc370d656f48319818"],
            "Dread Siege": ["120783f1fc7f2fd5e93221030"],
        },
        "red": {"Magma Skin": ["040443587c4f5390f46e11010"]},
        "yellow": {"Blood Tracer": ["080884d27abfafe7c18060180"],
                   "Living Volcano": ["08080411febfa7c1c0703f0fc"],}
    },
    "Sylvie": {
        "blue": {
            "Acrobat": ["080a8492faf7bdebe9242a020"],
            "Showtime": ["040643f8ef75dd73bcfe19010"],
            "Swift Shoft": ["0c08c678b2986d16318e21030"],
            "Innersight": ["0c0fc7898aecfb75191c3f030"],
            "Seize Cover": ["0404c23909475395f4f611010", "0809c4681a8ee733e9e223030"],
        },
        "green": {
            "Quick Tumble": ["000301a0ee5ed796848319818", "00060381ccbde73d090633030"]
        },
        "red": {
            "Spiritual Focus": [
                "040c42d9b63bc4d5e9ee33010",
                "08088490fc77ada3e1dc22020",
                "0c08c6d0b4b32c85e9fe33030",
            ]
        },
        "yellow": {"Wandering Hawk": ["0606e3edfd47d1f0fc1e03000"]},
    },
    "Wukong": {
        "blue": {"World Cracker": ["000f87f3fc63bce7f1fc3e000"]},
        "green": {
            "Inner Agility": ["0c018770fe9daf6be97422020", "0600c3b87f4ed735f4ba11010"],
            "Yaoguai-Slayer": ["060781e8fa7f1f57d48211010"],
        },
        "red": {"Petrificative": ["0405e26c194f53d0649b1f818", "1c1fc4d236bea728c1643e030"],
                "Indestructible": ["04044398ff5f97c1f4bc1f010"]},
        "yellow": {
            "Banish": ["060463dcfd4f5dc1f82618806"],
            "Sea Pillar": ["0e0fe61d8d46531180c020000", "1c1fcc3b1e8ca623018040000"],
            "Defy": ["08088612f29fa7e1e078380e0"],

        },
    },
    "Yukimura": {
        "red": {
            "Quick Ninjitsu": [
                "0606620d3f4753d5c4c319818",
                "0c0cc6195ec67395c58233030",
            ],
            "Shadow Assault": ["04064384697e1f1384e119010", "080d8711d2fcbe2701c436020"],
        },
        "blue": {
            "Cruelty": ["0404e1f9615c5713e4f21d010", "0404e2fc314c1703e4f11d018"],
            "Vault": ["080984323c97a3e8f93c3e020"],
            "Mortal Blade": ["040642e91d43d0f43c8e17010"],
            "Shadowseal": ["040fc4e9785a64d5a17a3f010", "0c0fc5d97a96ada1e1703f030"],
        },
        "green": {"Chasing Shuriken": ["080e6319ef5bd3b6ccfa1f010"]},
        "yellow": {"Bloodbath": ["0606630cf14f53f07c2e06800", "0c0cc619e2dee7e0f85c01000"]},
    },
}


ARTIFACTS = {'Alchemy Exercise': ['00400030783f0fc3601100201',
                                  '80400020783f0fc3601100205',
                                  '00400020783f0fc3601100205'],
            'All in!': ['004300c0781e0300c0300c201', '804300c0781e0300c0310c201'],
            'Anti-Bank': ['000001e0fc1f0783f07800001'],
            'Barrel Theory': ['804001e0781e4791e47900607', '804001e0781e0781e07900205'],
            'Blocking Exercise': ['804000d0fc1f0fd3c4a10060d', '804000d0fe1f0fc3c020002bf'],
            'Blood Trial': ['00400080200f87c1e0ec39201'],
            'Body Armor': ['000083f0fc3f0fc3f0fe1e000'],
            'Book of Chance': ['000003f0cc2d0843f0fc00000'],
            'Bounty Hunter': ['004201e07c3c0603c0b004201', '004301e07c3c0703c09004201'],
            'Boxing Glove': ['000403c0fc3f0fe3f07808000'],
            'Clover': ['00000370fc3f0f83f05c00000',
            '00000330fc3f0f83f07c00000',
            '00000370fc3f0f83f07c00000'],
            'Copy Ninja': ['00400004793f0fd1e4010060f', '80400000783f0fc1e4010020d'],
            'Credit Loans': ['004001e0fc3b0fc3f0fc00201'],
            'Death Race': ['804300c07c370bc334790c205', '800301e0fc250bc330780c20f'],
            'Deep Well': ['004003f0fc1f07c1e07800201'],
            'Demon Contract': ['804000c05c3f0fc3f07900201', '804000c05c3f0fc3f07800201'],
            'Destiny Spiral': ['804001e0cc2d0bc370790c205', '004001e0cc2d0bc3707900205'],
            'Early Bird': ['000000c0383f07c0f0f820001', '004000c0383f07c0f0f830001'],
            'Early Graduation': ['000003f0fc1e8303b0fc00001',
            '000003f0fc1e870330fc02001', '004001e0fc1f074360fc02000'],
            'Easter Egg': ['804000c0781e0781e0790c205', '804000c0781e0781e47900205'],
            'Faith Exercise': ['004000c0fc3f0300c07800201'],
            'Fencing Exercise': ['804000043d1c4713d4fd006ff', '87c000003c3c2703d0fc003ff'],
            'Follow the Tide': ['804000d07c1f4793f4f10061f'],
            'Free Throws': ['000001f0cc61986330f800001'],
            'Fur Shirt': ['000003f0fc1e0781e07800001', '000003f0fc1e0381e07800001'],
            'Gatling': ['004000301c2e1f03c1e020001',
                        '004000301c2f1f03c1e038001',
                        '004000301c2e1f03c1e020201',
                        '004000301c2e1f03c1e038001'],
            'Gaze of the Elder': ['004001e0781e0780c03000201'],
            'Gift Box': ['000001f07c1e0fc3f0fc00001'],
            'Gold Brick': ['00000000301e0783f1fe00001', '00000000301f0783f1fe01001'],
            'Heat Training': ['000002c07c330ce3f98600001'],
            'Inspiration': ['000000d0380c0783f0fc00001'],
            'Keep Hammering': ['00480100c0300fc3f0f82e201'],
            'Kong!': ['000783e0dc370fc3f0fc0b000', '000703e0dc3f0fc3f0fc0f000'],
            'Learn From Doing': ['000001f0f83e0f83e0fc0e001'],
            'Lifting Exercise': ['804003f0fc1c4310c47902607', '804003f07c0e0300c07900201'],
            'Loot Master': ['804003e0d83f0dc1807e01201', '804007e0913e4d91847d03603'],
            'Lucky Day': ['804000c0783f4fd1e4310061f', '804000c0793f4fd1e4310063f'],
            'Magic Book': ['000003f0fc3f0fc3f07800001', '004003f0fc3f0fc3f07800201'],
            'Meditation Exercise': ['004000c0303f0781f07800201', '004010c4fd3f4701c07100202'],
            'Mind Exercise': ['000000e07c1f07c3f0f000001'],
            'Night Owl': ['80400130fc3f4cd1e47900605', '80400130fc3f0cd1e47900205', '80c01364f93e4c93647100603'],
            'Pace Exercise': ['00004378fc3e0701c03802001'],
            'Polar Exercise': ['004002f0f01f0fc2e03c00201'],
            'Profitable': ['804000c008060307f1fd0c205'],
            'Protein Powder': ['000781e0fc3f0fc3f0fc00000'],
            'Reborn': ['804807a0b83c0783300c00205'],
            'Repair Exercise': ['00400330fc1e0703b0c400201'],
            'Secret Book': ['000083f0cc000fc3f0bc00001', '000003f0cc000fc3f0fc00001'],
            'Sneakers': ['004000303c1f0783c0e000201'],
            'Specialization': ['000301f0cc6d9b6330fc0c000'],
            'Start Spinning!': ['000003f0fc3f8fe3f0fc0c000', '000003f0fc3f0fc3f0fc00000'],
            'Talisman': ['004001e0fc3f0fc3f07800201'],
            'Taste Purple': ['004001e0fc1f0301e00000201',
                            '004001e0fc3f0301e40100401',
                            '804001e0fc3f0300e00100201'],
            'Warlock Exercise': ['804000c0780c0fc3f03100205'],
            'NONE': ['804303f4b44f9b621cfd0e687', '804313f4b56fd9721cfd1e601', '81c303f0b64e91621afc1e29f']
 }


HEROES = {'Alicia': ['000001c0703c0701c2fcff3fd', '000001c0703c070182fcff2fc', '000000c0383f0fc3f0fc0302d'],
          'Alloya': ['00004010083f1e6fbffdee7b9', '00000020383f0fffbfff9f67d'],
          'Asuka': ['000001c0781e0f83e0fc3b0ee', '000001c0701c0703e0fc3f0fe'],
          'Brynhild': ['040000c47d5f7ffbbc3f99e7f',
                       '04000184790df9bbbc7f9de7f',
                       ],
          'Cull': ['0b003205f3e7ddf7fd1e27887',
                   '8040008060301c4f73bcef3fd',
                   '8040008461107c5f77bced3b5',
                   ],
          'Emrald': ['00000000301c2f3fcff7fffff', '00000000782f1bceff97fffff'],
          'Hellsing': ['000000e03805cff3fcfcff27c', '000000603c0fc7f1fcfc7f27e'],
          'Jacquelyn': ['00400080300c0381f07c1f87f',],
          'James': ['00400000781e0781c07c1f27f',
                    '804000c0380e0781f87f07eff','80400000781e0781c07c1f27f',
          ],
          'Jenny': ['804000c0301c0781f07c0927f', '004000c0301c0781f07c0927f'],
          'Kay': ['000000403807c7f1fcff7ffff',
                  '004000c07b0f43f0dc3759ef7',
                  ],
          'Laila': ['80400000700c0300c0381f2fd',],
          'Malachite': ['00000000f63ecdf67dfe2f1fc',
                        '0000000038bf1bf7fdfe3e8fc',
                        '00000000f43ecdf67dfe3f1fc',
                        ],
          'Merlina': ['80400034fe1f83d0140d0060b', 'bfc01034ff1fe3d8160d0060b'],
          'Mina': ['004000c0381c0781f07c1f27f',
          '004000c0300c0381e07c1f2ff',
          '000000c0381e0781f07c1f07e', 'ffc010c4310c4711f47d1f7ff'],
          'Moriatee': ['87c01184611c4713f4f99e6f7',
          '80400184611c4713f4ff9e677',
          '80400180601c4713f0ff9e67d',],
          'Samuel': ['00000000781fbbfffcdf9e77d', '00000942f8ffbbffbffb1e47c'],
          'Sylvie': ['004000c0701c0783e0fc1f2ff', '804100c0300e439044513f7fd'],
          'Wukong': ['004001c0711e07c1f07e0e6c3', '184201c0711f07e1fc7f02641'],
          'Yukimura': ['80400000303e0f41ecb933eef'],
          'Bedivere': ['00000080301c0703f1fc3f3fe'],
          'Linh': ['00400100781e0701e09c8f37f'],
 }


GENRES = {'Crit': ['62188731ce3b8fe3f87a1d836',
                   '711c4318e6398fe1f87a0e83a'],
          'Evasion': ['d8fa368db3348d2741f87e1f8',
                      'd9f67e9da26c8f2f43d07c1f0',
                      'd9f67e9da26c8f2f43d07c1f0'],
          'Frost': ['e1e86210fc3f084a1b87e0f03',
                    '60f83a5ad6bfafc952d6e0d83',
                    'e0f83b5a5495254b5b8360d83'],
          'Heal': ['f1fc7f3fcff3f63584743d1f6'],
          'Health': ['799e27cdfb7edfb72dc320c93', '71be6fdbf6fdbf774d8320ca7'],
          'Innerfire': ['370dc731cc6719661986659b6', '1f0fc3b8ee318d6358c6358d2'],
          'Mech': ['3b0ec3f07c3f0fc3f07c1f078'],
          'Shield': ['1f47d1f47d1f47d1f4795e579', '9e2789e2789e2789e2789e278'],
          'Spell': ['1e0380c231ccf97e5f87738cc',
                    '0f0380e11966f93f4dc779ce6'],
          'Toxin': ['4e5394c51144511445114c531', 'def33ccf33ccf31cc731cc733'],
          'Vulnerable': ['65996759d6749a6699b665996', '34ccb33cdf34cd334cd337cdf'],
          'Weaponry': ['7bfffdf57d174780e0380e030', '7bdef7fd7d4f43c0701806038'],
          }


@lru_cache
def get_hash_to_hero_info():
    hash_to_hero_info = {}
    for hero_name, hashes in HEROES.items():
        for hash in hashes:
            hash_to_hero_info[hash] = Hero(hero_name, hash)
    
    return hash_to_hero_info


@lru_cache
def get_hash_to_artifact_info():
    hash_to_artifact_info = {}
    for artifact_name, hashes in ARTIFACTS.items():
        for hash in hashes:
            hash_to_artifact_info[hash] = Artifact(artifact_name, hash)
    
    return hash_to_artifact_info


@lru_cache
def get_hash_to_trait_info(hero=None, color=None):
    hash_to_trait_info = {}
    for hero_name, data in TRAITS.items():
        if hero and hero != hero_name:
            continue

        for color_name, data2 in data.items():  # TODO: update data/data2 names
            if color and color != color_name:
                continue

            for trait_name, hashes in data2.items():
                for hash in hashes:
                    hash_to_trait_info[hash] = Trait(trait_name, hero_name, hash, color_name)

    return hash_to_trait_info


@lru_cache
def get_hash_to_genre_info():
    hash_to_genre_info = {}
    for genre_nane, hashes in GENRES.items():
        for hash in hashes:
            hash_to_genre_info[hash] = Genre(genre_nane, hash)
    
    return hash_to_genre_info


def get_icon_hash(image, bbox):
    x, y, w, h = bbox
    icon = image[y:y+h, x:x+w]
    img = Image.fromarray(icon)
    icon_hash = imagehash.average_hash(img, hash_size=DEFAULT_HASH_SIZE)

    return icon_hash


def get_middle_section(genre_image, x_cutoff=0.2, y_cutoff=0.45):
    x_cutoff = max(0.0, min(0.5, x_cutoff))
    y_cutoff = max(0.0, min(0.5, y_cutoff))

    height, width = genre_image.shape[:2]
    x_start = int(width * x_cutoff)
    x_end = width - int(width * x_cutoff)
    y_start = int(height * y_cutoff)
    y_end = height - int(height * y_cutoff)

    return genre_image[y_start:y_end, x_start:x_end]


def get_genre_hash(image, bbox):
    """Get the genre hash by only using the middle section of the icon."""
    x, y, w, h = bbox
    icon = image[y:y+h, x:x+w]

    middle = get_middle_section(icon)
    # TODO: remove the grayscale conversion. imagehash.average_hash already converyts to grayscale
    # gray = cv2.cvtColor(middle, cv2.COLOR_BGR2GRAY)
    # img = Image.fromarray(gray)
    img = Image.fromarray(middle)
    genre_hash = imagehash.average_hash(img, hash_size=10)
    return genre_hash


def guess_icon_hash(image, bbox, icon_type, unknown_obj, hero=None, color=None):
    if icon_type not in ('hero', 'trait', 'artifact'):
        raise ValueError('Unknown icon type')

    if icon_type == 'artifact':
        hero2info_func  = get_hash_to_artifact_info
        hamming_threshold = HAMMING_ARTIFACT_THRESHOLD
    elif icon_type == 'hero':
        hero2info_func = get_hash_to_hero_info
        hamming_threshold = HAMMING_HERO_THRESHOLD
    elif icon_type == 'trait':
        hero2info_func = get_hash_to_trait_info
        hamming_threshold = HAMMING_TRAIT_THRESHOLD

    icon_hash = get_icon_hash(image, bbox)
    if icon_type == 'trait':
        hash2info = hero2info_func(hero, color)
    else:
        hash2info = hero2info_func()
    
    # TODO: a lot of returns peppered everywhere - reorganize
    # This logic is mostly used for traits (e.g. only one red trait for alicia as of 20250405)
    all_names = set([v.name for v in hash2info.values()])
    if len(all_names) == 1:
        # TODO: might not want to select an arbitary info? shouldn't matter too much though
        info =  list(hash2info.values())[0]
        reference_hash = imagehash.hex_to_hash(info.reference_hash)
        hamming = reference_hash - icon_hash

        return info, icon_hash, hamming

    hammings = defaultdict(list)
    for reference_hash, info in hash2info.items():
        reference_hash = imagehash.hex_to_hash(reference_hash)

        hamming = reference_hash - icon_hash
        if hamming < hamming_threshold:
            hammings[hamming].append(info)
    
    # TODO: reformat this, kinda ugly
    if not hammings:
        return unknown_obj, icon_hash, 9999

    # hamming_counts = {k:len(v) for k,v in hammings.items()}
    lowest_hamming = min(hammings.keys())
    lowest_infos = hammings[lowest_hamming]
    unique_names = set([i.name for i in lowest_infos])
    if len(unique_names) > 1:
        print(f'Found multiple hashes with hamming distance {lowest_hamming}: {lowest_infos} (icon hash: {icon_hash})')
        return unknown_obj, icon_hash, 9999

    return lowest_infos[0], icon_hash, lowest_hamming


guess_artifact_hash  = partial(guess_icon_hash, icon_type='artifact', unknown_obj=UNKNOWN_ARTIFACT)
guess_hero_hash = partial(guess_icon_hash, icon_type='hero', unknown_obj=UNKNOWN_HERO)
guess_trait_hash = partial(guess_icon_hash, icon_type='trait', unknown_obj=UNKNOWN_TRAIT)


# TODO: may want to merge with the generic function
def guess_genre_hash(image, bbox):
    icon_hash = get_genre_hash(image, bbox)
    hash2info = get_hash_to_genre_info()

    hammings = defaultdict(list)
    for reference_hash, info in hash2info.items():
        reference_hash = imagehash.hex_to_hash(reference_hash)

        hamming = reference_hash - icon_hash
        if hamming < HAMMING_GENRE_THRESHOLD:
            hammings[hamming].append(info)

    if not hammings:
        return UNKNOWN_GENRE, icon_hash, 9999

    lowest_hamming = min(hammings.keys())
    lowest_infos = hammings[lowest_hamming]
    unique_names = set([i.name for i in lowest_infos])
    if len(unique_names) > 1:
        print(f'Found multiple hashes with hamming distance {lowest_hamming}: {lowest_infos} (icon hash: {icon_hash})')
        return UNKNOWN_GENRE, icon_hash, 9999

    return lowest_infos[0], icon_hash, lowest_hamming
