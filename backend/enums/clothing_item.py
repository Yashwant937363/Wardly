from enum import Enum

class Category(str, Enum):
    # ---------- Tops ----------
    TSHIRT = "T-Shirt"
    SHIRT = "Shirt"
    POLO = "Polo Shirt"
    BLOUSE = "Blouse"
    TANK_TOP = "Tank Top"
    CAMISOLE = "Camisole"
    CROP_TOP = "Crop Top"
    TUNIC = "Tunic"
    SWEATER = "Sweater"
    CARDIGAN = "Cardigan"
    HOODIE = "Hoodie"
    SWEATSHIRT = "Sweatshirt"
    VEST = "Vest"

    # ---------- Outerwear ----------
    JACKET = "Jacket"
    BLAZER = "Blazer"
    COAT = "Coat"
    TRENCH_COAT = "Trench Coat"
    BOMBER_JACKET = "Bomber Jacket"
    DENIM_JACKET = "Denim Jacket"
    LEATHER_JACKET = "Leather Jacket"
    PUFFER_JACKET = "Puffer Jacket"

    # ---------- Bottoms ----------
    JEANS = "Jeans"
    TROUSERS = "Trousers"
    CHINOS = "Chinos"
    SHORTS = "Shorts"
    SKIRT = "Skirt"
    MINI_SKIRT = "Mini Skirt"
    MIDI_SKIRT = "Midi Skirt"
    MAXI_SKIRT = "Maxi Skirt"
    LEGGINGS = "Leggings"
    JOGGERS = "Joggers"
    SWEATPANTS = "Sweatpants"
    CARGO_PANTS = "Cargo Pants"

    # ---------- Dresses ----------
    DRESS = "Dress"
    MINI_DRESS = "Mini Dress"
    MIDI_DRESS = "Midi Dress"
    MAXI_DRESS = "Maxi Dress"
    BODYCON_DRESS = "Bodycon Dress"
    WRAP_DRESS = "Wrap Dress"
    SHIRT_DRESS = "Shirt Dress"
    GOWN = "Gown"

    # ---------- One Piece ----------
    JUMPSUIT = "Jumpsuit"
    ROMPER = "Romper"
    DUNGAREES = "Dungarees"

    # ---------- Ethnic ----------
    SAREE = "Saree"
    LEHENGA = "Lehenga"
    KURTA = "Kurta"
    KURTI = "Kurti"
    SHERWANI = "Sherwani"
    SALWAR = "Salwar"
    DUPATTA = "Dupatta"

    # ---------- Footwear ----------
    SNEAKERS = "Sneakers"
    RUNNING_SHOES = "Running Shoes"
    LOAFERS = "Loafers"
    BOOTS = "Boots"
    CHELSEA_BOOTS = "Chelsea Boots"
    SANDALS = "Sandals"
    FLATS = "Flats"
    HEELS = "Heels"
    WEDGES = "Wedges"
    SLIPPERS = "Slippers"

    # ---------- Accessories ----------
    BELT = "Belt"
    CAP = "Cap"
    HAT = "Hat"
    BEANIE = "Beanie"
    SCARF = "Scarf"
    GLOVES = "Gloves"
    TIE = "Tie"
    BOW_TIE = "Bow Tie"

class Closure(str, Enum):
    NONE = "None"
    BUTTONS = "Buttons"
    ZIPPER = "Zipper"
    SNAP = "Snap Buttons"
    VELCRO = "Velcro"
    DRAWSTRING = "Drawstring"
    ELASTIC = "Elastic"
    HOOK = "Hook"
    TIE = "Tie"
    BUCKLE = "Buckle"

class ColorFamily(str, Enum):
    BLACK = "Black"
    WHITE = "White"
    GREY = "Grey"

    BLUE = "Blue"
    NAVY = "Navy"

    GREEN = "Green"
    OLIVE = "Olive"

    RED = "Red"
    MAROON = "Maroon"

    YELLOW = "Yellow"
    ORANGE = "Orange"

    PURPLE = "Purple"
    LAVENDER = "Lavender"

    PINK = "Pink"

    BROWN = "Brown"
    BEIGE = "Beige"
    TAN = "Tan"

    GOLD = "Gold"
    SILVER = "Silver"

    MULTICOLOR = "Multicolor"
    NEUTRAL = "Neutral"

class Fabric(str, Enum):
    COTTON = "Cotton"
    LINEN = "Linen"
    DENIM = "Denim"
    WOOL = "Wool"
    SILK = "Silk"
    LEATHER = "Leather"
    FAUX_LEATHER = "Faux Leather"
    SATIN = "Satin"
    CHIFFON = "Chiffon"
    LACE = "Lace"
    VELVET = "Velvet"
    KNIT = "Knit"
    FLEECE = "Fleece"
    POLYESTER = "Polyester"
    NYLON = "Nylon"
    RAYON = "Rayon"
    VISCOSE = "Viscose"
    SPANDEX = "Spandex"
    UNKNOWN = "Unknown"


class Fit(str, Enum):
    SLIM = "Slim Fit"
    REGULAR = "Regular Fit"
    RELAXED = "Relaxed Fit"
    OVERSIZED = "Oversized"
    LOOSE = "Loose"

    SKINNY = "Skinny"
    STRAIGHT = "Straight"
    TAPERED = "Tapered"
    BOOTCUT = "Bootcut"
    FLARED = "Flared"
    WIDE_LEG = "Wide Leg"

    BODYCON = "Bodycon"
    A_LINE = "A-Line"
    SHIFT = "Shift"
    WRAP = "Wrap"
    FIT_AND_FLARE = "Fit and Flare"

class NeckStyle(str, Enum):
    NONE = "None"

    CREW = "Crew Neck"
    ROUND = "Round Neck"
    VNECK = "V-Neck"
    U_NECK = "U-Neck"
    SCOOP = "Scoop Neck"
    SQUARE = "Square Neck"
    BOAT = "Boat Neck"
    HALTER = "Halter Neck"
    OFF_SHOULDER = "Off Shoulder"
    ONE_SHOULDER = "One Shoulder"
    SWEETHEART = "Sweetheart Neck"
    TURTLENECK = "Turtleneck"

    POLO = "Polo Collar"
    BUTTON_DOWN = "Button Down"
    SPREAD = "Spread Collar"
    MANDARIN = "Mandarin Collar"
    SHAWL = "Shawl Collar"
    HOOD = "Hood"

class Pattern(str, Enum):
    SOLID = "Solid"
    STRIPED = "Striped"
    CHECKED = "Checked"
    PLAID = "Plaid"
    HOUNDSTOOTH = "Houndstooth"
    FLORAL = "Floral"
    GRAPHIC = "Graphic"
    PRINTED = "Printed"
    ABSTRACT = "Abstract"
    POLKA_DOT = "Polka Dot"
    CAMOUFLAGE = "Camouflage"
    ANIMAL_PRINT = "Animal Print"
    GEOMETRIC = "Geometric"
    PAISLEY = "Paisley"
    TIE_DYE = "Tie Dye"

class Texture(str, Enum):
    SMOOTH = "Smooth"
    RIBBED = "Ribbed"
    KNITTED = "Knitted"
    FLEECE = "Fleece"
    DENIM = "Denim"
    LEATHER = "Leather"
    LACE = "Lace"
    QUILTED = "Quilted"
    CROCHET = "Crochet"
    FUR = "Fur"

class SleeveLength(str, Enum):
    SLEEVELESS = "Sleeveless"
    CAP = "Cap Sleeve"
    SHORT = "Short Sleeve"
    ELBOW = "Elbow Sleeve"
    THREE_QUARTER = "Three Quarter"
    LONG = "Full Sleeve"