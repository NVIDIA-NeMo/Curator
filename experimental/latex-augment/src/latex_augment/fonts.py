# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import os
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache

if sys.platform == "darwin":
    GOOGLE_FONTS_DIR = os.path.expanduser("~/Library/texlive/2024/texmf-var/fonts/fonts-main")
else:
    GOOGLE_FONTS_DIR = "/usr/share/fonts/google_fonts"

# https://tug.org/FontCatalogue/

# commented-out fonts don't work in pdflatex (tested with texlive-full in Ubuntu 24.04.1)

PDFLATEX_FONTS = {
    "accanthis": "\\usepackage[T1]{fontenc}\n\\usepackage{accanthis}",
    "acorn": "\\input Acorn.fd\n\\newcommand*\\initfamily{\\usefont{U}{Acorn}{xl}{n}}",
    "alegreya": "\\usepackage[T1]{fontenc}\n\\usepackage{Alegreya}\n\\renewcommand*\\oldstylenums[1]{{\\AlegreyaOsF #1}}",
    "alegreyasans": "\\usepackage[T1]{fontenc}\n\\usepackage[sfdefault]{AlegreyaSans}\n\n\\renewcommand*\\oldstylenums[1]{{\\AlegreyaSansOsF #1}}",
    "alfaslabone": "\\usepackage{alfaslabone}\n\\usepackage[T1]{fontenc}",
    "algolrevived": "\\usepackage{algolrevived}\n\\usepackage[T1]{fontenc}",
    "almendra": "\\usepackage{almendra}\n\\usepackage[T1]{fontenc}",
    "annstone": "\\input AnnSton.fd\n\\newcommand*\\initfamily{\\usefont{U}{AnnSton}{xl}{n}}",
    "anonymouspro": "\\usepackage[T1]{fontenc}\n\\usepackage[ttdefault=true]{AnonymousPro}\n\\renewcommand*\\familydefault{\\ttdefault}",
    # "antiqua": '\\usepackage{antiqua}\n\\usepackage[T1]{fontenc}',
    "antykwapoltawskiego": "\\usepackage{antpolt}\n\\usepackage[T1]{fontenc}",
    "antykwapoltawskiegolight": "\\usepackage[light]{antpolt}\n\\usepackage[T1]{fontenc}",
    "antykwatorunska": "\\usepackage[math]{anttor}\n\\usepackage[T1]{fontenc}",
    "antykwatorunskacondensed": "\\usepackage[condensed,math]{anttor}\n\\usepackage[T1]{fontenc}",
    "antykwatorunskalight": "\\usepackage[light,math]{anttor}\n\\usepackage[T1]{fontenc}",
    "antykwatorunskalightcondensed": "\\usepackage[light,condensed,math]{anttor}\n\\usepackage[T1]{fontenc}",
    # "apicturealfabet": '\\usepackage{emerald}\n\\usepackage[T1]{fontenc}',
    "archivo": "\\usepackage{Archivo}\n\\usepackage[T1]{fontenc}",
    "arev": "\\usepackage{arev}\n\\usepackage[T1]{fontenc}",
    "arimo": "\\usepackage[sfdefault]{arimo}\n\\usepackage[T1]{fontenc}",
    "artificialuncial": "\\usepackage{auncial}\n\\usepackage[B1]{fontenc}",
    "artnouveaucaps": "\\input ArtNouvc.fd\n\\newcommand*\\initfamily{\\usefont{U}{ArtNouvc}{xl}{n}}",
    "artnouveauinitialen": "\\input ArtNouv.fd\n\\newcommand*\\initfamily{\\usefont{U}{ArtNouv}{xl}{n}}",
    "arvo": "\\usepackage{Arvo}\n\\usepackage[T1]{fontenc}",
    # "asanamath": '\\usepackage{fontspec}\n\\setmainfont{Asana-Math}',
    "ascii": "\\usepackage{ascii}\n\\usepackage[T1]{fontenc}",
    "atkinsonhyperlegible": "\\usepackage[sfdefault]{atkinson}\n\n\\usepackage[T1]{fontenc}",
    # "augie": '\\usepackage{emerald}\n\\usepackage[T1]{fontenc}',
    "auriocuskalligraphicus": "\\usepackage{aurical}\n\\usepackage[T1]{fontenc}",
    "baroqueinitials": "\\usepackage{yfonts}\n\\usepackage[T1]{fontenc}",
    "baskervaldadf": "\\usepackage{baskervald}\n\\usepackage[T1]{fontenc}",
    "baskervaldx": "\\usepackage[lf]{Baskervaldx}\n\\usepackage[bigdelims,vvarbb]{newtxmath}\n\\usepackage[cal=boondoxo]{mathalfa}\n\\renewcommand*\\oldstylenums[1]{\\textosf{#1}}",
    "baskervillef": "\\usepackage[T1]{fontenc}\n\\usepackage{baskervillef}\n\\usepackage[varqu,varl,var0]{inconsolata}\n\\usepackage[scale=.95,type1]{cabin}\n\\usepackage[baskerville,vvarbb]{newtxmath}\n\\usepackage[cal=boondoxo]{mathalfa}",
    "bbold": "\\usepackage{bbold}\n\\usepackage[T1]{fontenc}",
    "bera": "\\usepackage{bera}\n\n\\usepackage[T1]{fontenc}",
    "beramono": "\\usepackage[scaled]{beramono}\n\\renewcommand*\\familydefault{\\ttdefault}\n\\usepackage[T1]{fontenc}",
    "berasans": "\\usepackage[scaled]{berasans}\n\\renewcommand*\\familydefault{\\sfdefault}\n\\usepackage[T1]{fontenc}",
    "berenisadf": "\\usepackage{berenis}\n\\usepackage[LY1]{fontenc}",
    "beuron": "\\usepackage{beuron}",
    "biolinum": "\\usepackage[T1]{fontenc}\n\\usepackage[sfdefault]{biolinum}",
    "bitter": "\\usepackage{bitter}\n\\usepackage[T1]{fontenc}",
    "boisik": "\\usepackage{boisik}\n\\usepackage[OT1]{fontenc}",
    "bookman": "\\usepackage{bookman}\n\\usepackage[T1]{fontenc}",
    "brushscriptxitalic": "\\usepackage{pbsi}\n\\usepackage[T1]{fontenc}",
    "cabin": "\\usepackage[sfdefault]{cabin}\n\\usepackage[T1]{fontenc}",
    "cabincondensed": "\\usepackage[sfdefault,condensed]{cabin}\n\\usepackage[T1]{fontenc}",
    "caladea": "\\usepackage{caladea}\n\\usepackage[T1]{fontenc}",
    "calligra": "\\usepackage{calligra}\n\\usepackage[T1]{fontenc}",
    "cantarell": "\\usepackage[default]{cantarell}\n\\usepackage[T1]{fontenc}",
    "capitalbaseball": "\\usepackage{addfont}\n\\addfont{OT1}{capbas}{\\capbas}\n\\addfont{OT1}{capbasd}{\\capbasd}",
    "carlito": "\\usepackage[sfdefault,lf]{carlito}\n\n\\usepackage[T1]{fontenc}\n\\renewcommand*\\oldstylenums[1]{\\carlitoOsF #1}",
    "carolinganminuscules": "\\usepackage{carolmin}\n\\usepackage[T1]{fontenc}",
    "carrickcaps": "\\input Carrickc.fd\n\\newcommand*\\initfamily{\\usefont{U}{Carrickc}{xl}{n}}",
    "cascadiacode": "\\usepackage{cascadia-code}\n\\renewcommand*\\familydefault{\\ttdefault}\n\\usepackage[T1]{fontenc}",
    "charissil": "\\usepackage{CharisSIL}",
    "charterbt": "\\usepackage[bitstream-charter]{mathdesign}\n\\usepackage[T1]{fontenc}",
    "chivolight": "\\usepackage[familydefault,light]{Chivo}\n\\usepackage[T1]{fontenc}",
    "chivoregular": "\\usepackage[familydefault,regular]{Chivo}\n\\usepackage[T1]{fontenc}",
    "cinzel": "\\usepackage[default]{cinzel}\n\\usepackage[T1]{fontenc}",
    "clara": "\\usepackage{clara}\n\\usepackage[T1]{fontenc}",
    "clearsans": "\\usepackage[sfdefault]{ClearSans}\n\\usepackage[T1]{fontenc}",
    "cmpica": "\\usepackage{addfont}\n\\addfont{OT1}{cmpica}{\\pica}\n\\addfont{OT1}{cmpicab}{\\picab}\n\\addfont{OT1}{cmpicati}{\\picati}",
    "cochineal": "\\usepackage{cochineal}\n\\usepackage[T1]{fontenc}",
    "coelacanthextralight": "\\usepackage[el,nf]{coelacanth}\n\\usepackage[T1]{fontenc}\n\n\\let\\oldnormalfont\\normalfont\n\\def\\normalfont{\\oldnormalfont\\mdseries}",
    "coelacanthlight": "\\usepackage[l,nf]{coelacanth}\n\\usepackage[T1]{fontenc}\n\n\\let\\oldnormalfont\\normalfont\n\\def\\normalfont{\\oldnormalfont\\mdseries}",
    "coelacanthregular": "\\usepackage[nf]{coelacanth}\n\\usepackage[T1]{fontenc}\n\n\\let\\oldnormalfont\\normalfont\n\\def\\normalfont{\\oldnormalfont\\mdseries}",
    "comfortaa": "\\usepackage[default]{comfortaa}\n\\usepackage[T1]{fontenc}",
    "comicneue": "\\usepackage[default]{comicneue}\n\\usepackage[T1]{fontenc}",
    "comicneueangular": "\\usepackage[default,angular]{comicneue}\n\\usepackage[T1]{fontenc}",
    "computerconcrete": "\\usepackage{concmath}\n\\usepackage[OT1]{fontenc}",
    "computerconcreteeuler": "\\usepackage{beton}\n\\usepackage{euler}\n\\usepackage[OT1]{fontenc}",
    "computermodern": "\\usepackage[OT1]{fontenc}",
    "computermodernbright": "\\usepackage{cmbright}\n\\usepackage[OT1]{fontenc}",
    "computermoderndunhillroman": "\\renewcommand*\\rmdefault{cmdh}\n\\usepackage[OT1]{fontenc}",
    "computermodernfunnyroman": "\\renewcommand*\\rmdefault{cmfr}\n\\usepackage[OT1]{fontenc}",
    "computermoderngray": "\\usepackage{addfont}\n\\addfont{OT1}{cmgray}{\\cmgray}",
    "computermodernoutline": "\\usepackage[OT1]{fontenc}\n\\renewcommand*\\familydefault{ocm}",
    "computermodernromanfibonacci": "\\renewcommand*\\rmdefault{cmfib}\n\\usepackage[T1]{fontenc}",
    "computermodernsansserif": "\\usepackage[OT1]{fontenc}\n\\renewcommand*\\familydefault{\\sfdefault}",
    "computermodernsansserifoutline": "\\usepackage[OT1]{fontenc}\n\\renewcommand*\\familydefault{ocmss}",
    "computermodernsansserifquotation": "\\renewcommand*\\sfdefault{lcmss}\n\\renewcommand*\\familydefault{\\sfdefault}\n\\usepackage[OT1]{fontenc}",
    "computermodernteletype": "\\usepackage[OT1]{fontenc}\n\\renewcommand*\\familydefault{\\ttdefault}",
    "computermodernteletypequotation": "\\renewcommand*\\ttdefault{lcmtt}\n\\renewcommand*\\familydefault{\\ttdefault}\n\\usepackage[OT1]{fontenc}",
    "computermoderntypewriterproportional": "\\renewcommand*\\ttdefault{cmvtt}\n\\renewcommand*\\familydefault{\\ttdefault}\n\\usepackage[OT1]{fontenc}",
    "cormorantgaramond": "\\usepackage{CormorantGaramond}",
    "cormorantgaramondlight": "\\usepackage[light]{CormorantGaramond}",
    "courier": "\\usepackage{courier}\n\\renewcommand*\\familydefault{\\ttdefault}\n\\usepackage[T1]{fontenc}",
    "couriertenpitchbt": "\\usepackage{courierten}\n\\renewcommand*\\familydefault{\\ttdefault}\n\\usepackage[T1]{fontenc}",
    # "covington": '\\usepackage{covfonts}\n\\usepackage[T1]{fontenc}',
    "crimsonproextralight": "\\usepackage[extralight]{CrimsonPro}\n\\usepackage[T1]{fontenc}\n\n\\let\\oldnormalfont\\normalfont\n\\def\\normalfont{\\oldnormalfont\\mdseries}",
    "crimsonprolight": "\\usepackage[light]{CrimsonPro}\n\\usepackage[T1]{fontenc}\n\n\\let\\oldnormalfont\\normalfont\n\\def\\normalfont{\\oldnormalfont\\mdseries}",
    "crimsonpromedium": "\\usepackage[medium]{CrimsonPro}\n\\usepackage[T1]{fontenc}\n\n\\let\\oldnormalfont\\normalfont\n\\def\\normalfont{\\oldnormalfont\\mdseries}",
    "crimsonproregular": "\\usepackage{CrimsonPro}\n\\usepackage[T1]{fontenc}\n\n\\let\\oldnormalfont\\normalfont\n\\def\\normalfont{\\oldnormalfont\\mdseries}",
    "crimsontext": "\\usepackage{crimson}\n\\usepackage[T1]{fontenc}",
    "cuprum": "\\usepackage{cuprum}\n\\usepackage[T1]{fontenc}",
    "cyklop": "\\usepackage{cyklop}\n\\usepackage[T1]{fontenc}",
    "dayroman": "\\renewcommand*\\rmdefault{dayrom}\n\\usepackage[T1]{fontenc}",
    "dayromans": "\\renewcommand*\\rmdefault{dayroms}\n\\usepackage[T1]{fontenc}",
    # "decadence": '\\usepackage{emerald}\n\\usepackage[T1]{fontenc}',
    "dejavumono": "\\usepackage{DejaVuSansMono}\n\n\n\\renewcommand*\\familydefault{\\ttdefault}\n\\usepackage[T1]{fontenc}",
    "dejavusans": "\\usepackage{DejaVuSans}\n\n\n\\renewcommand*\\familydefault{\\sfdefault}\n\\usepackage[T1]{fontenc}",
    "dejavusanscondensed": "\\usepackage{DejaVuSansCondensed}\n\\renewcommand*\\familydefault{\\sfdefault}\n\\usepackage[T1]{fontenc}",
    "dejavuserif": "\\usepackage{DejaVuSerif}\n\\usepackage[T1]{fontenc}",
    "dejavuserifcondensed": "\\usepackage{DejaVuSerifCondensed}\n\\usepackage[T1]{fontenc}",
    "deutschenormalscrift": "\\usepackage{wedn}\n\\usepackage[T1]{fontenc}",
    "domitian": "\\usepackage{mathpazo}\n\\usepackage{domitian}\n\\usepackage[T1]{fontenc}\n\\let\\oldstylenums\\oldstyle",
    "drm": "\\usepackage[T1]{fontenc}\n\\usepackage{drm}",
    "droidsans": "\\usepackage[defaultsans]{droidsans}\n\\renewcommand*\\familydefault{\\sfdefault}\n\\usepackage[T1]{fontenc}",
    "droidsansmono": "\\usepackage[defaultmono]{droidsansmono}\n\\renewcommand*\\familydefault{\\ttdefault}\n\\usepackage[T1]{fontenc}",
    "droidserif": "\\usepackage[default]{droidserif}\n\\usepackage[T1]{fontenc}",
    "duerer": "\\usepackage{duerer}\n\\usepackage[T1]{fontenc}",
    "duererinformal": "\\usepackage{duerer}\n\\usepackage[T1]{fontenc}",
    "duerersansserif": "\\usepackage{duerer}\n\\usepackage[T1]{fontenc}",
    "duerertyperwriter": "\\usepackage{duerer}\n\\usepackage[T1]{fontenc}",
    "earlygothic": "\\usepackage{egothic}\n\\usepackage[T1]{fontenc}",
    "ebgaramond": "\\usepackage[cmintegrals,cmbraces]{newtxmath}\n\\usepackage{ebgaramond-maths}\n\\usepackage[T1]{fontenc}",
    "ebgaramonddecorativeinitials": "\\usepackage{ebgaramond}\n\\usepackage[T1]{fontenc}",
    # "eczarmedium": '\\usepackage{fontspec}\n\\setmainfont{Eczar-Medium.otf}[BoldFont=Eczar-Bold.otf]',
    # "eczarregular": '\\usepackage{fontspec}\n\\setmainfont{Eczar-Regular.otf}[BoldFont=Eczar-SemiBold.otf]',
    "eiad": "\\newcommand*\\eiadfamily{\\fontencoding{OT1}\\fontfamily{eiad}\\selectfont}\n\\usepackage[OT1]{fontenc}",
    "eiadconcrete": "\\newcommand*\\eiadcrfamily{\\fontencoding{OT1}\\fontfamily{eiadcc}\\selectfont}\n\\usepackage[OT1]{fontenc}",
    "eiadsansserif": "\\newcommand*\\eiadssfamily{\\fontencoding{OT1}\\fontfamily{eiadss}\\selectfont}\n\\usepackage[OT1]{fontenc}",
    "eiadtypewriter": "\\newcommand*\\eiadttfamily{\\fontencoding{OT1}\\fontfamily{eiadtt}\\selectfont}\n\\usepackage[OT1]{fontenc}",
    "eichenlaubinitialen": "\\input Eichenla.fd\n\\newcommand*\\initfamily{\\usefont{U}{Eichenla}{xl}{n}}",
    "eileencapsblack": "\\input EileenBl.fd\n\\newcommand*\\initfamily{\\usefont{U}{EileenBl}{xl}{n}}",
    "eileencapsregular": "\\input Eileen.fd\n\\newcommand*\\initfamily{\\usefont{U}{Eileen}{xl}{n}}",
    "electrumadf": "\\usepackage[lf]{electrum}\n\n\\usepackage[T1]{fontenc}",
    "elzeviercapsregular": "\\input Elzevier.fd\n\\newcommand*\\initfamily{\\usefont{U}{Elzevier}{xl}{n}}",
    "epigrafica": "\\usepackage{epigrafica}\n\\usepackage[LGR,OT1]{fontenc}",
    "erewhon": "\\usepackage[proportional,scaled=1.064]{erewhon}\n\\usepackage[erewhon,vvarbb,bigdelims]{newtxmath}\n\\usepackage[T1]{fontenc}\n\\renewcommand*\\oldstylenums[1]{\\textosf{#1}}",
    # "erewhon-math": '\\usepackage{unicode-math}\n\\setmathfont{Erewhon Math}\n\\setmainfont{Erewhon Regular}',
    "etbb": "\\usepackage[T1]{fontenc}\n\\usepackage{ETbb}\n\\let\\oldstylenums\\textosf",
    "europeancomputerconcrete": "\\usepackage{concmath}\n\\usepackage[T1]{fontenc}",
    "europeancomputerconcreteeuler": "\\usepackage{beton}\n\\usepackage{euler}\n\\usepackage[T1]{fontenc}",
    "europeancomputermodern": "\\usepackage[T1]{fontenc}",
    "europeancomputermodernbright": "\\usepackage{cmbright}\n\\usepackage[T1]{fontenc}",
    "europeancomputermoderndunhillroman": "\\renewcommand*\\rmdefault{cmdh}\n\\usepackage[T1]{fontenc}",
    "europeancomputermodernfunnyroman": "\\renewcommand*\\rmdefault{cmfr}\n\\usepackage[T1]{fontenc}",
    "europeancomputermodernsansserif": "\\usepackage[T1]{fontenc}\n\\renewcommand*\\familydefault{\\sfdefault}",
    "europeancomputermodernsansserifquotation": "\\renewcommand*\\sfdefault{lcmss}\n\\renewcommand*\\familydefault{\\sfdefault}\n\\usepackage[T1]{fontenc}",
    "europeancomputermodernteletype": "\\usepackage[T1]{fontenc}\n\\renewcommand*\\familydefault{\\ttdefault}",
    "europeancomputermodernteletypequotation": "\\renewcommand*\\ttdefault{lcmtt}\n\\renewcommand*\\familydefault{\\ttdefault}\n\\usepackage[T1]{fontenc}",
    "europeancomputermoderntypewriterproportional": "\\renewcommand*\\ttdefault{cmvtt}\n\\renewcommand*\\familydefault{\\ttdefault}\n\\usepackage[T1]{fontenc}",
    "fbb": "\\usepackage[T1]{fontenc}\n\\usepackage{fbb}",
    "fetamont": "\\usepackage{fetamont}\n\\usepackage[T1]{fontenc}",
    "firamono": "\\usepackage{FiraMono}\n\\renewcommand*\\familydefault{\\ttdefault}\n\\usepackage[T1]{fontenc}",
    "firasansbook": "\\usepackage[sfdefault,book]{FiraSans}\n\\usepackage[T1]{fontenc}\n\\renewcommand*\\oldstylenums[1]{{\\firaoldstyle #1}}",
    "firasansextralight": "\\usepackage[sfdefault,extralight]{FiraSans}\n\\usepackage[T1]{fontenc}\n\\renewcommand*\\oldstylenums[1]{{\\firaoldstyle #1}}",
    "firasanslight": "\\usepackage[sfdefault,light]{FiraSans}\n\\usepackage[T1]{fontenc}\n\\renewcommand*\\oldstylenums[1]{{\\firaoldstyle #1}}",
    # "firasansmath": '\\usepackage[sfdefault,lining]{FiraSans}\n\\usepackage[fakebold]{firamath-otf}\n\\renewcommand*\\oldstylenums[1]{{\\firaoldstyle #1}}',
    "firasansnewtxsf": "\\usepackage[T1]{fontenc}\n\\usepackage[sfdefault,scaled=.85]{FiraSans}\n\\usepackage{newtxsf}",
    "firasansregular": "\\usepackage[sfdefault]{FiraSans}\n\\usepackage[T1]{fontenc}\n\\renewcommand*\\oldstylenums[1]{{\\firaoldstyle #1}}",
    "firasansthin": "\\usepackage[sfdefault,thin]{FiraSans}\n\\usepackage[T1]{fontenc}\n\\renewcommand*\\oldstylenums[1]{{\\firaoldstyle #1}}",
    "firasansultralight": "\\usepackage[sfdefault,ultralight]{FiraSans}\n\\usepackage[T1]{fontenc}\n\\renewcommand*\\oldstylenums[1]{{\\firaoldstyle #1}}",
    "flyspec": "\\usepackage{addfont}\n\\addfont{OT1}{flyspec}{\\flyspec}",
    "foekfont": "\\usepackage{foekfont}\n\\usepackage[T1]{fontenc}",
    "forum": "\\usepackage{forum}\n\\usepackage[T1]{fontenc}",
    "fraktur": "\\usepackage{yfonts}\n\\usepackage[T1]{fontenc}",
    "frenchcursive": "\\usepackage[default]{frcursive}\n\\usepackage[T1]{fontenc}",
    # "gandhisans": '\\usepackage[lf,sfdefault]{gandhi}\n\\usepackage[T1]{fontenc}',
    # "gandhiserif": '\\usepackage[lf]{gandhi}\n\\usepackage[T1]{fontenc}',
    # "garamond": '\\usepackage[urw-garamond]{mathdesign}\n\\usepackage[T1]{fontenc}',
    # "garamondexpertmathdesign": '\\usepackage[T1]{fontenc}\n\\usepackage[urw-garamond]{mathdesign}\n\\usepackage{garamondx}',
    # "garamondexpertnewtxmath": '\\usepackage[T1]{fontenc}\n\\usepackage{garamondx}\n\\usepackage[garamondx,cmbraces]{newtxmath}',
    # "garamondlibre": '\\usepackage{fontspec}\n\\setmainfont{Garamond Libre}',
    "gentium": "\\usepackage[T1]{fontenc}\n\\usepackage{gentium}",
    "gfsartemisia": "\\usepackage{gfsartemisia}\n\\usepackage[T1]{fontenc}",
    "gfsartemisiaeulermath": "\\usepackage{gfsartemisia-euler}\n\\usepackage[T1]{fontenc}",
    "gfsbodoni": "\\usepackage[default]{gfsbodoni}\n\\usepackage[T1]{fontenc}",
    "gfsdidot": "\\usepackage{gfsdidot}\n\\usepackage[T1]{fontenc}",
    "gfsneohellenic": "\\usepackage[default]{gfsneohellenic}\n\\usepackage[LGR,T1]{fontenc}",
    # "gfsneohellenicmath": '\\usepackage{gfsneohellenicot}',
    "gillius": "\\usepackage[T1]{fontenc}\n\\usepackage[default]{gillius}",
    # "gnufreefontsans": '\\usepackage{fontspec}\n\\setmainfont{FreeSans}',
    # "gnufreefontserif": '\\usepackage{fontspec}\n\\setmainfont{FreeSerif}',
    "gomono": "\\usepackage{GoMono}\n\\renewcommand*\\familydefault{\\ttdefault}\n\\usepackage[T1]{fontenc}",
    "gosans": "\\usepackage[sfdefault]{GoSans}\n\\usepackage[T1]{fontenc}",
    "gothictexturaprescius": "\\usepackage{pgothic}\n\\usepackage[T1]{fontenc}",
    "gothictexturaquadrata": "\\usepackage[T1]{fontenc}\n\\usepackage{tgothic}",
    "gotik": "\\usepackage{yfonts}\n\\usepackage[T1]{fontenc}",
    "gotischeinitialen": "\\input GotIn.fd\n\\newcommand*\\initfamily{\\usefont{U}{GotIn}{xl}{n}}",
    "goudyinitialen": "\\input GoudyIn.fd\n\\newcommand*\\initfamily{\\usefont{U}{GoudyIn}{xl}{n}}",
    "gudea": "\\usepackage{Gudea}\n\\usepackage[T1]{fontenc}",
    "halfuncial": "\\usepackage{huncial}\n\\usepackage[T1]{fontenc}",
    "hersheyoldenglishfont": "\\usepackage{addfont}\n\\addfont[1.25]{OT1}{hge}{\\hge}",
    "heuristica": "\\usepackage{heuristica}\n\\usepackage[heuristica,vvarbb,bigdelims]{newtxmath}\n\\usepackage[T1]{fontenc}\n\\renewcommand*\\oldstylenums[1]{\\textosf{#1}}",
    "hindmaduraimedium": "\\usepackage[medium]{HindMadurai}\n\\usepackage[T1]{fontenc}",
    "hindmadurairegular": "\\usepackage[semibold]{HindMadurai}\n\\usepackage[T1]{fontenc}",
    "humanist": "\\usepackage{humanist}\n\\usepackage[T1]{fontenc}",
    "ibarrarealnova": "\\usepackage[lining]{ibarra}\n\\usepackage[T1]{fontenc}",
    # "ibmplexmonoextralight": '\\usepackage[T1]{fontenc}\n\\usepackage[usefilenames,RMstyle=ExtraLight,SSstyle=ExtraLight,TTstyle=ExtraLight,DefaultFeatures={Ligatures=Common}]{plex-otf}\n\\renewcommand*\\familydefault{\\ttdefault}',
    # "ibmplexmonolight": '\\usepackage[T1]{fontenc}\n\\usepackage[usefilenames,RMstyle=Light,SSstyle=Light,TTstyle=Light,DefaultFeatures={Ligatures=Common}]{plex-otf}\n\\renewcommand*\\familydefault{\\ttdefault}',
    # "ibmplexmonomedium": '\\usepackage[T1]{fontenc}\n\\usepackage[usefilenames,RMstyle={Medium,Semibold},SSstyle={Medium,Semibold},TTstyle={Medium,Semibold},DefaultFeatures={Ligatures=Common}]{plex-otf}\n\\renewcommand*\\familydefault{\\ttdefault}',
    # "ibmplexmonoregular": '\\usepackage[T1]{fontenc}\n\\usepackage[usefilenames,DefaultFeatures={Ligatures=Common}]{plex-otf}\n\\renewcommand*\\familydefault{\\ttdefault}',
    # "ibmplexmonotext": '\\usepackage[T1]{fontenc}\n\\usepackage[usefilenames,RMstyle={Text,Semibold},SSstyle={Text,Semibold},TTstyle={Text,Semibold},DefaultFeatures={Ligatures=Common}]{plex-otf}\n\\renewcommand*\\familydefault{\\ttdefault}',
    # "ibmplexmonothin": '\\usepackage[T1]{fontenc}\n\\usepackage[usefilenames,RMstyle=Thin,SSstyle=Thin,TTstyle=Thin,DefaultFeatures={Ligatures=Common}]{plex-otf}\n\\renewcommand*\\familydefault{\\ttdefault}',
    # "ibmplexsansextralight": '\\usepackage[T1]{fontenc}\n\\usepackage[usefilenames,RMstyle=ExtraLight,SSstyle=ExtraLight,TTstyle=ExtraLight,DefaultFeatures={Ligatures=Common}]{plex-otf}\n\\renewcommand*\\familydefault{\\sfdefault}',
    # "ibmplexsanslight": '\\usepackage[T1]{fontenc}\n\\usepackage[usefilenames,RMstyle=Light,SSstyle=Light,TTstyle=Light,DefaultFeatures={Ligatures=Common}]{plex-otf}\n\\renewcommand*\\familydefault{\\sfdefault}',
    # "ibmplexsansmedium": '\\usepackage[T1]{fontenc}\n\\usepackage[usefilenames,RMstyle={Medium,Semibold},SSstyle={Medium,Semibold},TTstyle={Medium,Semibold},DefaultFeatures={Ligatures=Common}]{plex-otf}\n\\renewcommand*\\familydefault{\\sfdefault}',
    # "ibmplexsansregular": '\\usepackage[T1]{fontenc}\n\\usepackage[usefilenames,DefaultFeatures={Ligatures=Common}]{plex-otf}\n\\renewcommand*\\familydefault{\\sfdefault}',
    # "ibmplexsanstext": '\\usepackage[T1]{fontenc}\n\\usepackage[usefilenames,RMstyle={Text,Semibold},SSstyle={Text,Semibold},TTstyle={Text,Semibold},DefaultFeatures={Ligatures=Common}]{plex-otf}\n\\renewcommand*\\familydefault{\\sfdefault}',
    # "ibmplexsansthin": '\\usepackage[T1]{fontenc}\n\\usepackage[usefilenames,RMstyle=Thin,SSstyle=Thin,TTstyle=Thin,DefaultFeatures={Ligatures=Common}]{plex-otf}\n\\renewcommand*\\familydefault{\\sfdefault}',
    # "ibmplexserifextralight": '\\usepackage[T1]{fontenc}\n\\usepackage[usefilenames,RMstyle=ExtraLight,SSstyle=ExtraLight,TTstyle=ExtraLight,DefaultFeatures={Ligatures=Common}]{plex-otf}',
    # "ibmplexseriflight": '\\usepackage[T1]{fontenc}\n\\usepackage[usefilenames,RMstyle=Light,SSstyle=Light,TTstyle=Light,DefaultFeatures={Ligatures=Common}]{plex-otf}',
    # "ibmplexserifmedium": '\\usepackage[T1]{fontenc}\n\\usepackage[usefilenames,RMstyle={Medium,Semibold},SSstyle={Medium,Semibold},TTstyle={Medium,Semibold},DefaultFeatures={Ligatures=Common}]{plex-otf}',
    # "ibmplexserifregular": '\\usepackage[T1]{fontenc}\n\\usepackage[usefilenames,DefaultFeatures={Ligatures=Common}]{plex-otf}',
    # "ibmplexseriftext": '\\usepackage[T1]{fontenc}\n\\usepackage[usefilenames,RMstyle={Text,Semibold},SSstyle={Text,Semibold},TTstyle={Text,Semibold},DefaultFeatures={Ligatures=Common}]{plex-otf}',
    # "ibmplexserifthin": '\\usepackage[T1]{fontenc}\n\\usepackage[usefilenames,RMstyle=Thin,SSstyle=Thin,TTstyle=Thin,DefaultFeatures={Ligatures=Common}]{plex-otf}',
    "imfellenglish": "\\usepackage{imfellEnglish}\n\\usepackage[T1]{fontenc}",
    "inconsolata": "\\usepackage{inconsolata}\n\\renewcommand*\\familydefault{\\ttdefault}\n\\usepackage[T1]{fontenc}",
    "inriasanslight": "\\usepackage[T1]{fontenc}\n\\usepackage[lining,light]{InriaSans}\n\\renewcommand*\\oldstylenums[1]{{\\fontfamily{InriaSans-OsF}\\selectfont #1}}\n\n\\let\\oldnormalfont\\normalfont\n\\def\\normalfont{\\oldnormalfont\\mdseries}",
    "inriasansregular": "\\usepackage[T1]{fontenc}\n\\usepackage[lining]{InriaSans}\n\\renewcommand*\\oldstylenums[1]{{\\fontfamily{InriaSans-OsF}\\selectfont #1}}\n\n\\let\\oldnormalfont\\normalfont\n\\def\\normalfont{\\oldnormalfont\\mdseries}",
    "inriaseriflight": "\\usepackage[T1]{fontenc}\n\\usepackage[lining,light]{InriaSerif}\n\\renewcommand*\\oldstylenums[1]{{\\fontfamily{InriaSerif-OsF}\\selectfont #1}}\n\n\\let\\oldnormalfont\\normalfont\n\\def\\normalfont{\\oldnormalfont\\mdseries}",
    "inriaserifregular": "\\usepackage[T1]{fontenc}\n\\usepackage[lining]{InriaSerif}\n\\renewcommand*\\oldstylenums[1]{{\\fontfamily{InriaSerif-OsF}\\selectfont #1}}\n\n\\let\\oldnormalfont\\normalfont\n\\def\\normalfont{\\oldnormalfont\\mdseries}",
    "insularmajuscule": "\\usepackage{inslrmaj}\n\\usepackage[T1]{fontenc}",
    "insularminuscule": "\\usepackage{inslrmin}\n\\usepackage[T1]{fontenc}",
    "interextralight": "\\usepackage[sfdefault,extralight]{inter}\n\n\\usepackage[T1]{fontenc}",
    "interlight": "\\usepackage[sfdefault,light]{inter}\n\n\\usepackage[T1]{fontenc}",
    "intermedium": "\\usepackage[sfdefault,medium]{inter}\n\n\\usepackage[T1]{fontenc}",
    "interregular": "\\usepackage[sfdefault]{inter}\n\n\\usepackage[T1]{fontenc}",
    "interthin": "\\usepackage[sfdefault,thin]{inter}\n\n\\usepackage[T1]{fontenc}",
    # "intimacy": '\\usepackage{emerald}\n\\usepackage[T1]{fontenc}',
    "iwona": "\\usepackage[math]{iwona}\n\\usepackage[T1]{fontenc}",
    "iwonacondensed": "\\usepackage[condensed,math]{iwona}\n\\usepackage[T1]{fontenc}",
    "iwonalight": "\\usepackage[light,math]{iwona}\n\\usepackage[T1]{fontenc}",
    "iwonalightcondensed": "\\usepackage[light,condensed,math]{iwona}\n\\usepackage[T1]{fontenc}",
    "janaskrivana": "\\usepackage{aurical}\n\\usepackage[T1]{fontenc}",
    # "jd": '\\usepackage{emerald}\n\\usepackage[T1]{fontenc}',
    "josefinsans": "\\usepackage[sfdefault]{josefin}\n\\usepackage[T1]{fontenc}",
    "josefinsansextralight": "\\usepackage[sfdefault,extralight,medium]{josefin}\n\\usepackage[T1]{fontenc}",
    "josefinsanslight": "\\usepackage[sfdefault,light,medium]{josefin}\n\\usepackage[T1]{fontenc}",
    "josefinsansthin": "\\usepackage[sfdefault,thin,medium]{josefin}\n\\usepackage[T1]{fontenc}",
    # "junicode": '\\usepackage{fontspec}\n\\setmainfont{Junicode}[\n  Extension=.ttf,\n  BoldFont=*-Bold]',
    "kerkis": "\\usepackage{kmath,kerkis}\n\\usepackage[T1]{fontenc}",
    "kinigsteincaps": "\\input Kinigcap.fd\n\\newcommand*\\initfamily{\\usefont{U}{Kinigcap}{xl}{n}}",
    "konanurkaps": "\\input Konanur.fd\n\\newcommand*\\initfamily{\\usefont{U}{Konanur}{xl}{n}}",
    "kpmonospaced": "\\usepackage{kpfonts}\n\\renewcommand*\\familydefault{\\ttdefault}\n\\usepackage[T1]{fontenc}",
    "kpsansserif": "\\usepackage[sfmath]{kpfonts}\n\\renewcommand*\\familydefault{\\sfdefault}\n\\usepackage[T1]{fontenc}",
    "kpserif": "\\usepackage{kpfonts}\n\\usepackage[T1]{fontenc}",
    "kramerregular": "\\input Kramer.fd\n\\newcommand*\\initfamily{\\usefont{U}{Kramer}{xl}{n}}",
    "kurier": "\\usepackage[math]{kurier}\n\\usepackage[T1]{fontenc}",
    "kuriercondensed": "\\usepackage[condensed,math]{kurier}\n\\usepackage[T1]{fontenc}",
    "kurierlight": "\\usepackage[light,math]{kurier}\n\\usepackage[T1]{fontenc}",
    "kurierlightcondensed": "\\usepackage[light,condensed,math]{kurier}\n\\usepackage[T1]{fontenc}",
    "lateinischeausgangsschrift": "\\usepackage{wela}\n\\usepackage[T1]{fontenc}",
    "latinmodernmono": "\\usepackage{lmodern}\n\\renewcommand*\\familydefault{\\ttdefault}\n\\usepackage[T1]{fontenc}",
    "latinmodernmonolight": "\\usepackage{lmodern}\n\\renewcommand*\\familydefault{\\ttdefault}\n\\usepackage[T1]{fontenc}",
    "latinmodernmonolightcondensed": "\\usepackage{lmodern}\n\\renewcommand*\\familydefault{\\ttdefault}\n\\usepackage[T1]{fontenc}",
    "latinmodernmonoproportional": "\\renewcommand*\\ttdefault{lmvtt}\n\\renewcommand*\\familydefault{\\ttdefault}\n\\usepackage[T1]{fontenc}",
    "latinmodernmonoz": "\\usepackage[scaled=1.05,proportional,lightcondensed]{zlmtt}\n\\renewcommand*\\familydefault{\\ttdefault}\n\\usepackage[T1]{fontenc}",
    "latinmodernroman": "\\usepackage{lmodern}\n\\usepackage[T1]{fontenc}",
    "latinmodernroman-ae": "\\usepackage{lmodern}\n\\usepackage[T1]{fontenc}\n\\usepackage{aesupp}",
    "latinmodernsans": "\\usepackage{lmodern}\n\\renewcommand*\\familydefault{\\sfdefault}\n\\usepackage[T1]{fontenc}",
    "latinmodernsansextended": "\\renewcommand*\\sfdefault{lmssq}\n\\renewcommand*\\familydefault{\\sfdefault}\n\\usepackage[T1]{fontenc}",
    "lato": "\\usepackage[default]{lato}\n\\usepackage[T1]{fontenc}",
    # "lettergothic": '\\usepackage[scaled]{ulgothic}\n\\renewcommand*\\familydefault{\\ttdefault}\n\\usepackage[T1]{fontenc}',
    # "lexend": '\\usepackage{lexend}',
    "libertinusmono": "\\usepackage{libertinus}\n\\usepackage[T1]{fontenc}\n\\renewcommand*\\familydefault{\\ttdefault}",
    "libertinussans": "\\usepackage{libertinus}\n\\usepackage[T1]{fontenc}\n\\renewcommand*\\familydefault{\\sfdefault}",
    "libertinusserif": "\\usepackage{libertinus}\n\\usepackage[T1]{fontenc}",
    "librebaskerville": "\\usepackage{librebaskerville}\n\\usepackage[T1]{fontenc}",
    "librebodoni": "\\usepackage[T1]{fontenc}\n\\usepackage{LibreBodoni}",
    "librecaslon": "\\usepackage{librecaslon}\n\\usepackage[T1]{fontenc}",
    "librefranklin": "\\usepackage[T1]{fontenc}\n\\usepackage{librefranklin}\n\\renewcommand*\\familydefault{\\sfdefault}",
    "librisadf": "\\usepackage{libris}\n\\renewcommand*\\familydefault{\\sfdefault}\n\\usepackage[T1]{fontenc}",
    "linuxlibertine": "\\usepackage{libertine}\n\\usepackage{libertinust1math}\n\\usepackage[T1]{fontenc}",
    # "literaturnaya": '\\usepackage{literat}\n\\usepackage[T1]{fontenc}',
    "lobstertwo": "\\usepackage{LobsterTwo}\n\\usepackage[T1]{fontenc}",
    "logo": "\\usepackage{mflogo}\n\\usepackage[T1]{fontenc}",
    "lukassvatba": "\\usepackage{aurical}\n\\usepackage[T1]{fontenc}",
    # "luximono": '\\usepackage{luximono}\n\\renewcommand*\\familydefault{\\ttdefault}\n\\usepackage[T1]{fontenc}',
    "lxfonts": "\\usepackage[T1]{fontenc}\n\\usepackage{lxfonts}",
    "magra": "\\usepackage{Magra}\n\\usepackage[T1]{fontenc}",
    # "malvern": '\\input T1fmv.fd\n\\renewcommand*\\sfdefault{fmv}\n\\renewcommand*\\familydefault{\\sfdefault}\n\\usepackage[T1]{fontenc}',
    "marcellus": "\\usepackage{marcellus}\n\\usepackage[T1]{fontenc}",
    "merriweather": "\\usepackage{merriweather}\n\\usepackage[T1]{fontenc}",
    "merriweatherlight": "\\usepackage[light]{merriweather}",
    "merriweathersans": "\\usepackage[sfdefault]{merriweather}\n\\usepackage[T1]{fontenc}",
    "merriweathersanslight": "\\usepackage[sfdefault,light]{merriweather}",
    "miamanueva": "\\usepackage{miama}\n\\usepackage[T1]{fontenc}",
    "mintspirit": "\\usepackage[T1]{fontenc}\n\\usepackage[default]{mintspirit}",
    # "missaali": '\\usepackage{fontspec}\n\\setmainfont{Missaali-Regular.otf}',
    "mlmodern": "\\usepackage{mlmodern}\n\\usepackage[T1]{fontenc}",
    "montserratalternatesextralight": "\\usepackage[defaultfam,extralight,tabular,lining,alternates]{montserrat}\n\n\\usepackage[T1]{fontenc}\n\\renewcommand*\\oldstylenums[1]{{\\fontfamily{Montserrat-TOsF}\\selectfont #1}}",
    "montserratalternateslight": "\\usepackage[defaultfam,light,tabular,lining,alternates]{montserrat}\n\n\\usepackage[T1]{fontenc}\n\\renewcommand*\\oldstylenums[1]{{\\fontfamily{Montserrat-TOsF}\\selectfont #1}}",
    "montserratalternatesregular": "\\usepackage[defaultfam,tabular,lining,alternates]{montserrat}\n\n\\usepackage[T1]{fontenc}\n\\renewcommand*\\oldstylenums[1]{{\\fontfamily{Montserrat-TOsF}\\selectfont #1}}",
    "montserratalternatesthin": "\\usepackage[defaultfam,thin,tabular,lining,alternates]{montserrat}\n\n\\usepackage[T1]{fontenc}\n\\renewcommand*\\oldstylenums[1]{{\\fontfamily{Montserrat-TOsF}\\selectfont #1}}",
    "montserratextralight": "\\usepackage[defaultfam,extralight,tabular,lining]{montserrat}\n\n\\usepackage[T1]{fontenc}\n\\renewcommand*\\oldstylenums[1]{{\\fontfamily{Montserrat-TOsF}\\selectfont #1}}",
    "montserratlight": "\\usepackage[defaultfam,light,tabular,lining]{montserrat}\n\n\\usepackage[T1]{fontenc}\n\\renewcommand*\\oldstylenums[1]{{\\fontfamily{Montserrat-TOsF}\\selectfont #1}}",
    "montserratregular": "\\usepackage[defaultfam,tabular,lining]{montserrat}\n\n\\usepackage[T1]{fontenc}\n\\renewcommand*\\oldstylenums[1]{{\\fontfamily{Montserrat-TOsF}\\selectfont #1}}",
    "montserratthin": "\\usepackage[defaultfam,thin,tabular,lining]{montserrat}\n\n\\usepackage[T1]{fontenc}\n\\renewcommand*\\oldstylenums[1]{{\\fontfamily{Montserrat-TOsF}\\selectfont #1}}",
    "morrisinitialen": "\\input MorrisIn.fd\n\\newcommand*\\initfamily{\\usefont{U}{MorrisIn}{xl}{n}}",
    # "movieola": '\\usepackage{emerald}\n\\usepackage[T1]{fontenc}',
    "necker": "\\usepackage{addfont}\n\\addfont{OT1}{necker}{\\necker}",
    # "newcomputermodernroman": '\\usepackage{fontspec}\n\\usepackage{unicode-math}\n\n\\setmainfont[\nItalicFont=NewCM10-Italic.otf,\nBoldFont=NewCM10-Bold.otf,\nBoldItalicFont=NewCM10-BoldItalic.otf,\nSmallCapsFeatures={Numbers=OldStyle}]{NewCM10-Regular.otf}\n\n\\setsansfont[\nItalicFont=NewCMSans10-Oblique.otf,\nBoldFont=NewCMSans10-Bold.otf,\nBoldItalicFont=NewCMSans10-BoldOblique.otf,\nSmallCapsFeatures={Numbers=OldStyle}]{NewCMSans10-Regular.otf}\n\n\\setmonofont[ItalicFont=NewCMMono10-Italic.otf,\nBoldFont=NewCMMono10-Bold.otf,\nBoldItalicFont=NewCMMono10-BoldOblique.otf,\nSmallCapsFeatures={Numbers=OldStyle}]{NewCMMono10-Regular.otf}\n\n\\setmathfont{NewCMMath-Regular.otf}',
    # "newcomputermodernsansserif": '\\usepackage{fontspec}\n\\usepackage{unicode-math}\n\n\\setmainfont[\nItalicFont=NewCM10-Italic.otf,\nBoldFont=NewCM10-Bold.otf,\nBoldItalicFont=NewCM10-BoldItalic.otf,\nSmallCapsFeatures={Numbers=OldStyle}]{NewCM10-Regular.otf}\n\n\\setsansfont[\nItalicFont=NewCMSans10-Oblique.otf,\nBoldFont=NewCMSans10-Bold.otf,\nBoldItalicFont=NewCMSans10-BoldOblique.otf,\nSmallCapsFeatures={Numbers=OldStyle}]{NewCMSans10-Regular.otf}\n\n\\setmonofont[ItalicFont=NewCMMono10-Italic.otf,\nBoldFont=NewCMMono10-Bold.otf,\nBoldItalicFont=NewCMMono10-BoldOblique.otf,\nSmallCapsFeatures={Numbers=OldStyle}]{NewCMMono10-Regular.otf}\n\n\\setmathfont{NewCMMath-Regular.otf}\n\n\\renewcommand*\\familydefault{\\sfdefault}',
    # "newcomputermoderntypewriter": '\\usepackage{fontspec}\n\\usepackage{unicode-math}\n\n\\setmainfont[\nItalicFont=NewCM10-Italic.otf,\nBoldFont=NewCM10-Bold.otf,\nBoldItalicFont=NewCM10-BoldItalic.otf,\nSmallCapsFeatures={Numbers=OldStyle}]{NewCM10-Regular.otf}\n\n\\setsansfont[\nItalicFont=NewCMSans10-Oblique.otf,\nBoldFont=NewCMSans10-Bold.otf,\nBoldItalicFont=NewCMSans10-BoldOblique.otf,\nSmallCapsFeatures={Numbers=OldStyle}]{NewCMSans10-Regular.otf}\n\n\\setmonofont[ItalicFont=NewCMMono10-Italic.otf,\nBoldFont=NewCMMono10-Bold.otf,\nBoldItalicFont=NewCMMono10-BoldOblique.otf,\nSmallCapsFeatures={Numbers=OldStyle}]{NewCMMono10-Regular.otf}\n\n\\setmathfont{NewCMMath-Regular.otf}\n\n\\renewcommand*\\familydefault{\\ttdefault}',
    "newpx": "\\usepackage[T1]{fontenc}\n\\usepackage{newpxtext,newpxmath}",
    "newpxeuler": "\\usepackage[T1]{fontenc}\n\\usepackage{newpxtext,eulerpx}",
    "newtx": "\\usepackage[T1]{fontenc}\n\\usepackage{newtxtext,newtxmath}",
    "newtxtt": "\\usepackage[T1]{fontenc}\n\\usepackage[zerostyle=d]{newtxtt}\n\\renewcommand*\\familydefault{\\ttdefault}",
    "nimbus15mono": "\\usepackage{nimbusmono}\n\\renewcommand*\\familydefault{\\ttdefault}\n\\usepackage[T1]{fontenc}",
    "nimbus15mononarrow": "\\usepackage{nimbusmononarrow}\n\\renewcommand*\\familydefault{\\ttdefault}\n\\usepackage[T1]{fontenc}",
    "nimbus15sans": "\\usepackage{nimbussans}\n\\renewcommand*\\familydefault{\\sfdefault}\n\\usepackage[T1]{fontenc}",
    "nimbus15serif": "\\usepackage{nimbusserif}\n\\usepackage[T1]{fontenc}",
    "notomono": "\\usepackage{noto}\n\\usepackage[T1]{fontenc}\n\\renewcommand*\\familydefault{\\ttdefault}",
    "notosans": "\\usepackage[sfdefault]{noto}\n\\usepackage[T1]{fontenc}",
    "notosansmono": "\\usepackage{noto}\n\\usepackage[T1]{fontenc}\n\\renewcommand*\\familydefault{\\ttdefault}",
    "notoserif": "\\usepackage{notomath}\n\\usepackage[T1]{fontenc}",
    "nouveaudropcaps": "\\input Nouveaud.fd\n\\newcommand*\\initfamily{\\usefont{U}{Nouveaud}{xl}{n}}",
    "nunito": "\\usepackage{nunito}\n\\usepackage[T1]{fontenc}",
    "obyknovennayanovaya": "\\renewcommand*\\rmdefault{obn}\n\\usepackage[LCYW]{fontenc}",
    "ocr-a": "\\usepackage[ocr-a]{ocr}\n\\usepackage[T1]{fontenc}",
    "ocr-b": "\\usepackage{ocr}\n\\usepackage[T1]{fontenc}",
    # "oesterreichischeschulschrift": '\\usepackage{oesch}',
    "oldlatin": "\\usepackage{addfont}\n\\addfont[1.25]{OT1}{olr10}{\\olr}\n\\addshape{bx}{n}{olb10}",
    "oldstandard": "\\usepackage{OldStandard}\n\\usepackage[T1]{fontenc}",
    "opensans": "\\usepackage[default,oldstyle,scale=0.95]{opensans}\n\n\n\\usepackage[T1]{fontenc}",
    "oswald": "\\usepackage{Oswald}\n\\usepackage[T1]{fontenc}",
    "overlock": "\\usepackage[sfdefault]{overlock}\n\\usepackage[T1]{fontenc}",
    "pacioli": "\\usepackage{pacioli}\n\\usepackage[OT1]{fontenc}",
    # "pandora": '\\usepackage{pandora}\n\\usepackage[OT1]{fontenc}',
    "pandorasansserif": "\\usepackage{pandora}\n\\usepackage[T1]{fontenc}",
    "pandorasinglepitch": "\\usepackage{pandora}\n\\usepackage[T1]{fontenc}",
    "paratypesans": "\\usepackage{paratype}\n\\renewcommand*\\familydefault{\\sfdefault}\n\\usepackage[T1]{fontenc}",
    "paratypesanscaption": "\\usepackage{PTSansCaption} \n\\renewcommand*\\familydefault{\\sfdefault}\n\\usepackage[T1]{fontenc}",
    "paratypesansnarrow": "\\usepackage{PTSansNarrow} \n\\renewcommand*\\familydefault{\\sfdefault}\n\\usepackage[T1]{fontenc}",
    "paratypeserif": "\\usepackage{paratype} \n\\usepackage[T1]{fontenc}",
    "paratypeserifcaption": "\\usepackage{PTSerifCaption} \n\\usepackage[T1]{fontenc}",
    # "petevanroosescript": '\\usepackage{pvscript}\n\\usepackage[T1]{fontenc}',
    # "play": '\\usepackage{Play}\n\\usepackage[T1]{fontenc}',
    "playfairdisplay": "\\usepackage{PlayfairDisplay}\n\n\n\\renewcommand*\\oldstylenums[1]{{\\playfairOsF #1}}\n\\usepackage[T1]{fontenc}",
    "poiretone": "\\usepackage{PoiretOne}\n\\usepackage[T1]{fontenc}\n\\renewcommand*\\familydefault{\\sfdefault}",
    # "pookie": '\\usepackage{emerald}\n\\usepackage[T1]{fontenc}',
    "punk": "\\usepackage{punk}\n\\usepackage[T1]{fontenc}",
    # "punknova": '\\usepackage{fontspec}\n\\setmainfont{Punk Nova}',
    "pxfonts": "\\usepackage{pxfonts}\n\\usepackage[T1]{fontenc}",
    # "qtabbie": '\\usepackage{fontspec}\n\\setmainfont{QTAbbie}',
    # "qtagatetype": '\\usepackage{fontspec}\n\\setmainfont{QTAgateType}',
    # "qtancientolive": '\\usepackage{fontspec}\n\\setmainfont{QTAncientOlive}',
    # "qtantiquepost": '\\usepackage{fontspec}\n\\setmainfont{QTAntiquePost}',
    # "qtarabian": '\\usepackage{fontspec}\n\\setmainfont{QTArabian}',
    # "qtarnieb": '\\usepackage{fontspec}\n\\setmainfont{QTArnieB}',
    # "qtartiston": '\\usepackage{fontspec}\n\\setmainfont{QTArtiston}',
    # "qtatchen": '\\usepackage{fontspec}\n\\setmainfont{QTAtchen}',
    # "qtavanti": '\\usepackage{fontspec}\n\\setmainfont{QTAvanti}',
    # "qtbasker": '\\usepackage{fontspec}\n\\setmainfont{QTBasker}',
    # "qtbeckman": '\\usepackage{fontspec}\n\\setmainfont{QTBeckman}',
    # "qtbengal": '\\usepackage{fontspec}\n\\setmainfont{QTBengal}',
    # "qtblackforest": '\\usepackage{fontspec}\n\\setmainfont{QTBlackForest}',
    # "qtblimpo": '\\usepackage{fontspec}\n\\setmainfont{QTBlimpo}',
    # "qtbodini": '\\usepackage{fontspec}\n\\setmainfont{QTBodini}',
    # "qtbodiniposter": '\\usepackage{fontspec}\n\\setmainfont{QTBodiniPoster}',
    # "qtbookmann": '\\usepackage{fontspec}\n\\setmainfont{QTBookmann}',
    # "qtboulevard": '\\usepackage{fontspec}\n\\setmainfont{QTBoulevard}',
    # "qtbrushstroke": '\\usepackage{fontspec}\n\\setmainfont{QTBrushStroke}',
    # "qtcaligulatype": '\\usepackage{fontspec}\n\\setmainfont{QTCaligulatype}',
    # "qtcanaithtype": '\\usepackage{fontspec}\n\\setmainfont{QTCanaithtype}',
    # "qtcascadetype": '\\usepackage{fontspec}\n\\setmainfont{QTCascadetype}',
    # "qtcaslan": '\\usepackage{fontspec}\n\\setmainfont{QTCaslan}',
    # "qtcaslanopen": '\\usepackage{fontspec}\n\\setmainfont{QTCaslanOpen}',
    # "qtcasual": '\\usepackage{fontspec}\n\\setmainfont{QTCasual}',
    # "qtchancerytype": '\\usepackage{fontspec}\n\\setmainfont{QTChanceryType}',
    # "qtchicagoland": '\\usepackage{fontspec}\n\\setmainfont{QTChicagoland}',
    # "qtclaytablet": '\\usepackage{fontspec}\n\\setmainfont{QTClaytablet}',
    # "qtcloisteredmonk": '\\usepackage{fontspec}\n\\setmainfont{QTCloisteredMonk}',
    # "qtcoronation": '\\usepackage{fontspec}\n\\setmainfont{QTCoronation}',
    # "qtdeuce": '\\usepackage{fontspec}\n\\setmainfont{QTDeuce}',
    # "qtdoghaus": '\\usepackage{fontspec}\n\\setmainfont[BoldFont=*Heavy]{QTDoghaus}',
    # "qtdublinirish": '\\usepackage{fontspec}\n\\setmainfont{QTDublinIrish}',
    # "qteratype": '\\usepackage{fontspec}\n\\setmainfont{QTEraType}',
    # "qteurotype": '\\usepackage{fontspec}\n\\setmainfont{QTEurotype}',
    # "qtfloraline": '\\usepackage{fontspec}\n\\setmainfont{QTFloraline}',
    # "qtflorencia": '\\usepackage{fontspec}\n\\setmainfont{QTFlorencia}',
    # "qtfraktur": '\\usepackage{fontspec}\n\\setmainfont{QTFraktur}',
    # "qtfrank": '\\usepackage{fontspec}\n\\setmainfont[BoldFont=*Heavy]{QTFrank}',
    # "qtfrizquad": '\\usepackage{fontspec}\n\\setmainfont{QTFrizQuad}',
    # "qtfuture": '\\usepackage{fontspec}\n\\setmainfont{QTFuture}',
    # "qtfutureposter": '\\usepackage{fontspec}\n\\setmainfont{QTFuturePoster}',
    # "qtgaromand": '\\usepackage{fontspec}\n\\setmainfont{QTGaromand}',
    # "qtghoulface": '\\usepackage{fontspec}\n\\setmainfont{QTGhoulFace}',
    # "qtgraphlite": '\\usepackage{fontspec}\n\\setmainfont{QTGraphLite}',
    # "qtgraveure": '\\usepackage{fontspec}\n\\setmainfont{QTGraveure}',
    # "qtgreece": '\\usepackage{fontspec}\n\\setmainfont{QTGreece}',
    # "qthandwriting": '\\usepackage{fontspec}\n\\setmainfont{QTHandwriting}',
    # "qtheidelbergtype": '\\usepackage{fontspec}\n\\setmainfont{QTHeidelbergType}',
    # "qthelvetblack": '\\usepackage{fontspec}\n\\setmainfont{QTHelvet-Black}',
    # "qthelvetboldoutline": '\\usepackage{fontspec}\n\\setmainfont{QTHelvet-BoldOutline}',
    # "qthelvetcnd": '\\usepackage{fontspec}\n\\setmainfont[BoldFont=*-Black]{QTHelvetCnd}',
    # "qthoboken": '\\usepackage{fontspec}\n\\setmainfont{QTHoboken}',
    # "qthowardtype": '\\usepackage{fontspec}\n\\setmainfont[BoldFont=*Fat]{QTHowardType}',
    # "qtimpromptu": '\\usepackage{fontspec}\n\\setmainfont{QTImpromptu}',
    # "qtjupiter": '\\usepackage{fontspec}\n\\setmainfont{QTJupiter}',
    # "qtkooper": '\\usepackage{fontspec}\n\\setmainfont{QTKooper}',
    # "qtkorrin": '\\usepackage{fontspec}\n\\setmainfont{QTKorrin}',
    # "qtkungfu": '\\usepackage{fontspec}\n\\setmainfont{QTKung-Fu}',
    # "qtlautrectype": '\\usepackage{fontspec}\n\\setmainfont{QTLautrecType}',
    # "qtlettergoth": '\\usepackage{fontspec}\n\\setmainfont{QTLetterGoth}',
    # "qtlinoscroll": '\\usepackage{fontspec}\n\\setmainfont{QTLinoscroll}',
    # "qtlinostroke": '\\usepackage{fontspec}\n\\setmainfont{QTLinostroke}',
    # "qtlondonscroll": '\\usepackage{fontspec}\n\\setmainfont{QTLondonScroll}',
    # "qtmagicmarker": '\\usepackage{fontspec}\n\\setmainfont{QTMagicMarker}',
    # "qtmerryscript": '\\usepackage{fontspec}\n\\setmainfont{QTMerryScript}',
    # "qtmilitary": '\\usepackage{fontspec}\n\\setmainfont{QTMilitary}',
    # "qtokcorral": '\\usepackage{fontspec}\n\\setmainfont{QTOKCorral}',
    # "qtokcorralcnd": '\\usepackage{fontspec}\n\\setmainfont{QTOKCorral-Cnd}',
    # "qtokcorralext": '\\usepackage{fontspec}\n\\setmainfont{QTOKCorral-Ext}',
    # "qtoldgoudy": '\\usepackage{fontspec}\n\\setmainfont{QTOldGoudy}',
    # "qtoptimum": '\\usepackage{fontspec}\n\\setmainfont{QTOptimum}',
    # "qtpalatine": '\\usepackage{fontspec}\n\\setmainfont{QTPalatine}',
    # "qtpandora": '\\usepackage{fontspec}\n\\setmainfont{QTPandora}',
    # "qtparisfrance": '\\usepackage{fontspec}\n\\setmainfont{QTParisFrance}',
    # "qtpeignoir": '\\usepackage{fontspec}\n\\setmainfont{QTPeignoir}',
    # "qtpeignoirlite": '\\usepackage{fontspec}\n\\setmainfont{QTPeignoir-Lite}',
    # "qtpiltdown": '\\usepackage{fontspec}\n\\setmainfont{QTPiltdown}',
    # "qtpristine": '\\usepackage{fontspec}\n\\setmainfont{QTPristine}',
    # "qtrobotic2000": '\\usepackage{fontspec}\n\\setmainfont{QTRobotic2000}',
    # "qtsandiego": '\\usepackage{fontspec}\n\\setmainfont{QTSanDiego}',
    # "qtschoolcentury": '\\usepackage{fontspec}\n\\setmainfont{QTSchoolCentury}',
    # "qtslogantype": '\\usepackage{fontspec}\n\\setmainfont{QTSlogantype}',
    # "qtsnowcaps": '\\usepackage{fontspec}\n\\setmainfont{QTSnowCaps}',
    # "qtstorytimecaps": '\\usepackage{fontspec}\n\\setmainfont{QTStoryTimeCaps}',
    # "qttechtone": '\\usepackage{fontspec}\n\\setmainfont{QTTechtone}',
    # "qttheatre": '\\usepackage{fontspec}\n\\setmainfont{QTTheatre}',
    # "qttimeoutline": '\\usepackage{fontspec}\n\\setmainfont{QTTimeOutline}',
    # "qttumbleweed": '\\usepackage{fontspec}\n\\setmainfont{QTTumbleweed}',
    # "qtusauncial": '\\usepackage{fontspec}\n\\setmainfont{QTUSA-Uncial}',
    # "qtvagaround": '\\usepackage{fontspec}\n\\setmainfont{QTVagaRound}',
    # "qtweise": '\\usepackage{fontspec}\n\\setmainfont{QTWeise}',
    # "qtwestend": '\\usepackage{fontspec}\n\\setmainfont{QTWestEnd}',
    "quattrocento": "\\usepackage{quattrocento}\n\\usepackage[T1]{fontenc}",
    "quattrocentosans": "\\usepackage[sfdefault]{quattrocento}\n\\usepackage[T1]{fontenc}",
    "raleway": "\\usepackage[T1]{fontenc}\n\\usepackage[default]{raleway}",
    "ralphsmithsformalscript": "\\usepackage{addfont}\n\\addfont{OT1}{rsfs10}{\\rsfs}",
    "roboto": "\\usepackage[sfdefault]{roboto}\n\\usepackage[T1]{fontenc}",
    "robotocondensed": "\\usepackage[sfdefault,condensed]{roboto}\n\\usepackage[T1]{fontenc}",
    "robotolight": "\\usepackage[sfdefault,light]{roboto}\n\\usepackage[T1]{fontenc}",
    "robotolightcondensed": "\\usepackage[sfdefault,light,condensed]{roboto}\n\\usepackage[T1]{fontenc}",
    "robotoslab": "\\usepackage[rm]{roboto}\n\\usepackage[T1]{fontenc}",
    "robotoslablight": "\\usepackage[rm,light]{roboto}\n\\usepackage[T1]{fontenc}",
    "robotoslabthin": "\\usepackage[rm,thin]{roboto}\n\\usepackage[T1]{fontenc}",
    "robotothin": "\\usepackage[sfdefault,thin]{roboto}\n\\usepackage[T1]{fontenc}",
    "romandeadf": "\\usepackage{romande}\n\\usepackage[T1]{fontenc}",
    "romanrustic": "\\usepackage{rustic}\n\\usepackage[T1]{fontenc}",
    "romantik": "\\input Romantik.fd\n\\newcommand*\\initfamily{\\usefont{U}{Romantik}{xl}{n}}",
    "rosario": "\\usepackage[familydefault]{Rosario}\n\\usepackage[T1]{fontenc}",
    "rothenburgdecorative": "\\input Rothdn.fd\n\\newcommand*\\initfamily{\\usefont{U}{Rothdn}{xl}{n}}",
    "rotunda": "\\usepackage{rotunda}\n\\usepackage[T1]{fontenc}",
    "royalinitialen": "\\input RoyalIn.fd\n\\newcommand*\\initfamily{\\usefont{U}{RoyalIn}{xl}{n}}",
    "sanremo": "\\input Sanremo.fd\n\\newcommand*\\initfamily{\\usefont{U}{Sanremo}{xl}{n}}",
    "sansmathfonts": "\\usepackage{sansmathfonts}\n\\usepackage[T1]{fontenc}\n\\renewcommand*\\familydefault{\\sfdefault}",
    "scholax": "\\usepackage[p,osf]{scholax}\n\n\\usepackage{amsmath,amsthm}\n\n\\usepackage[scaled=1.075,ncf,vvarbb]{newtxmath}",
    "schulausgangschrift": "\\usepackage{wesa}\n\\usepackage[T1]{fontenc}",
    "schwabacher": "\\usepackage{yfonts}\n\\usepackage[T1]{fontenc}",
    "schwell": "\\usepackage{suetterl}\n\\usepackage[T1]{fontenc}",
    "segmentfont": "\\usepackage{addfont}\n\\addfont{OT1}{d7seg}{\\dviiseg}\n\\addfont{OT1}{deseg}{\\deseg}",
    # "shobhika": '\\usepackage{fontspec}\n\\setmainfont{Shobhika}',
    "simfon": "\\usepackage{addfont}\n\\addfont[1.5]{OT1}{simfon}{\\simfon}",
    # "skeetch": '\\usepackage{emerald}\n\\usepackage[T1]{fontenc}',
    "slabikar": "\\usepackage{addfont}\n\\addfont[1.5]{OT1}{slabikar}{\\slabikar}",
    "sourcecodepro": "\\usepackage[default]{sourcecodepro}\n\\usepackage[T1]{fontenc}",
    "sourcesanspro": "\\usepackage[default]{sourcesanspro}\n\\usepackage[T1]{fontenc}",
    "sourceserifproextralight": "\\usepackage[default,extralight,semibold]{sourceserifpro}\n\\usepackage[T1]{fontenc}",
    "sourceserifprolight": "\\usepackage[default,light,bold]{sourceserifpro}\n\\usepackage[T1]{fontenc}",
    "sourceserifproregular": "\\usepackage[default,regular,black]{sourceserifpro}\n\\usepackage[T1]{fontenc}",
    # "spankysbungalow": '\\usepackage{emerald}\n\\usepackage[T1]{fontenc}',
    "spectral": "\\usepackage[t,lf]{spectral}\n\\usepackage[T1]{fontenc}",
    "spectrallight": "\\usepackage[t,lf,l]{spectral}\n\\usepackage[T1]{fontenc}",
    "squarecapitals": "\\usepackage{sqrcaps}\n\\usepackage[T1]{fontenc}",
    "starburstregular": "\\input Starburst.fd\n\\newcommand*\\initfamily{\\usefont{U}{Starburst}{xl}{n}}",
    "step": "\\usepackage[notext]{stix}\n\\usepackage{step}\n\\usepackage[T1]{fontenc}",
    "stickstoo": "\\usepackage{stickstootext}\n\\usepackage[stickstoo,vvarbb]{newtxmath}",
    "stix": "\\usepackage[T1]{fontenc}\n\\usepackage{stix}",
    "stix2": "\\usepackage[T1]{fontenc}\n\\usepackage{stix2}",
    "suetterlin": "\\usepackage{wesu}\n\\usepackage[T1]{fontenc}",
    # "tallpaul": '\\usepackage{emerald}\n\\usepackage[T1]{fontenc}',
    "tapir": "\\usepackage{addfont}\n\\addfont{OT1}{tap}{\\tapir}",
    # "teenspirit": '\\usepackage{emerald}\n\\usepackage[T1]{fontenc}',
    "texgyreadventor": "\\usepackage{tgadventor}\n\\renewcommand*\\familydefault{\\sfdefault}\n\\usepackage[T1]{fontenc}",
    "texgyrebonum": "\\usepackage{tgbonum}\n\\usepackage[T1]{fontenc}",
    "texgyrebonum-ae": "\\usepackage{tgbonum}\n\\usepackage[T1]{fontenc}\n\\usepackage{aesupp}",
    "texgyrechorus": "\\usepackage{tgchorus}\n\\usepackage[T1]{fontenc}",
    "texgyrecursor": "\\usepackage{tgcursor}\n\\renewcommand*\\familydefault{\\ttdefault}\n\\usepackage[T1]{fontenc}",
    "texgyreheros": "\\usepackage{tgheros}\n\\renewcommand*\\familydefault{\\sfdefault}\n\\usepackage[T1]{fontenc}",
    "texgyrepagella": "\\usepackage{tgpagella}\n\\usepackage[T1]{fontenc}",
    "texgyrepagella-ae": "\\usepackage{tgpagella}\n\\usepackage[T1]{fontenc}\n\\usepackage{aesupp}",
    "texgyreschola": "\\usepackage{tgschola}\n\\usepackage[T1]{fontenc}",
    "texgyreschola-ae": "\\usepackage{tgschola}\n\\usepackage[T1]{fontenc}\n\\usepackage{aesupp}",
    "texgyretermes": "\\usepackage{tgtermes}\n\\usepackage[T1]{fontenc}",
    "theanodidot": "\\usepackage{TheanoDidot}\n\\usepackage[T1]{fontenc}",
    "theanomodern": "\\usepackage{TheanoModern}\n\\usepackage[T1]{fontenc}",
    "theanooldstyle": "\\usepackage{TheanoOldStyle}\n\\usepackage[T1]{fontenc}",
    "tinos": "\\usepackage[T1]{fontenc}\n\\usepackage{tinos}",
    "trajan": "\\usepackage{trajan}\n\\usepackage[T1]{fontenc}",
    # "twcal14": '\\usepackage{twcal}\n\\usepackage[T1]{fontenc}',
    "txfonts": "\\usepackage{txfonts}\n\\usepackage[T1]{fontenc}",
    "txtt": "\\renewcommand*\\ttdefault{txtt}\n\\renewcommand*\\familydefault{\\ttdefault}\n\\usepackage[T1]{fontenc}",
    "typographercaps": "\\input Typocaps.fd\n\\newcommand*\\initfamily{\\usefont{U}{Typocaps}{xl}{n}}",
    "uncial": "\\usepackage{uncial}\n\\usepackage[T1]{fontenc}",
    "universal": "\\renewcommand*\\familydefault{uni}\n\\usepackage[OT1]{fontenc}",
    "universalisadfcondensed": "\\usepackage[condensed,sfdefault]{universalis}\n\\usepackage[T1]{fontenc}",
    "universalisadfstandard": "\\usepackage[sfdefault]{universalis}\n\\usepackage[T1]{fontenc}",
    # "urwa030": '\\usepackage[scaled]{uarial}\n\\renewcommand*\\familydefault{\\sfdefault}\n\\usepackage[T1]{fontenc}',
    "urwchancery": "\\usepackage{chancery}\n\\usepackage[T1]{fontenc}",
    # "urwclassico": '\\usepackage[sfdefault]{classico}\n\\usepackage[T1]{fontenc}',
    "urwgothic": "\\usepackage{avant}\n\\renewcommand*\\familydefault{\\sfdefault}\n\\usepackage[T1]{fontenc}",
    "urwgrotesk": "\\renewcommand*\\sfdefault{ugq}\n\\usepackage[T1]{fontenc}",
    "urwnimbusroman": "\\usepackage{mathptmx}\n\\usepackage[T1]{fontenc}",
    "urwnimbussans": "\\usepackage[scaled]{helvet}\n\\renewcommand*\\familydefault{\\sfdefault}\n\\usepackage[T1]{fontenc}",
    "urwpalladio": "\\usepackage[sc]{mathpazo}\n\\linespread{1.05}\n\\usepackage[T1]{fontenc}",
    "urwschoolbookl": "\\usepackage{fouriernc}\n\\usepackage[T1]{fontenc}",
    "utopia-fouriermath": "\\usepackage{fourier}\n\\usepackage[T1]{fontenc}",
    "utopia-mathdesign": "\\usepackage[adobe-utopia]{mathdesign}\n\\usepackage[T1]{fontenc}",
    "venturisadf": "\\usepackage[lf]{venturis}\n\n\\usepackage[T1]{fontenc}",
    "venturisadfno2": "\\usepackage{venturis2}\n\\usepackage[T1]{fontenc}",
    "venturisadfold": "\\usepackage{venturisold}\n\\usepackage[T1]{fontenc}",
    "venturisadfsans": "\\usepackage[lf]{venturis}\n\n\\renewcommand*\\familydefault{\\sfdefault}\n\\usepackage[T1]{fontenc}",
    "vereinfachteausgangsschrift": "\\usepackage{weva}",
    # "vicentino": '\\usepackage{vicent}\n\\usepackage[OT1]{fontenc}',
    # "webster": '\\usepackage{emerald}\n\\usepackage[T1]{fontenc}',
    "xcharter": "\\usepackage{XCharter}\n\\usepackage[T1]{fontenc}",
    # "xits": '\\usepackage{unicode-math}\n\\setmainfont{XITS}\n\\setmathfont{XITS Math}',
    "zallmancaps": "\\input Zallman.fd\n\\newcommand*\\initfamily{\\usefont{U}{Zallman}{xl}{n}}",
}


@dataclass
class Font:
    name: str  # font name (postscript syntax)
    path: str  # font file path
    index: int  # index within the TTC font collection file
    family: str  # font family name
    familylangs: list[str]  # font family language(s)

    @property
    def charset(self) -> str:
        return get_charset(self.path, self.index)

    @property
    def dirname(self) -> str:
        return os.path.dirname(self.path)

    @property
    def filename(self) -> str:
        return os.path.basename(self.path)

    def fontspec(self, cmd: str) -> str:
        """Construct XeLaTeX fontspec font selection command."""
        if self.path.endswith(".ttc"):
            return (
                f"{cmd}{{{self.filename}}}"
                f"["
                f"Path={self.dirname}/,"
                f"UprightFeatures={{FontIndex={self.index}}},"
                f"BoldFeatures={{FontIndex={self.index}}},"
                f"ItalicFeatures={{FontIndex={self.index}}},"
                f"BoldItalicFeatures={{FontIndex={self.index}}},"
                f"]"
            )
        else:
            return (
                f"{cmd}{{{self.filename}}}"
                f"["
                f"Path={self.dirname}/,"
                f"BoldFont={self.filename},"
                f"ItalicFont={self.filename},"
                f"BoldItalicFont={self.filename},"
                f"]"
            )

    def __hash__(self) -> int:
        """Make fonts hashable by PostScript name."""
        return hash(self.name)

    def __lt__(self, other: "Font") -> bool:
        """Make fonts sortable by PostScript name."""
        return self.name < other.name


@lru_cache(maxsize=None)
def list_fonts(pattern: str = ":") -> list[Font]:
    """List all fonts in the system.
    :returns: list of installed fonts
    """
    res = subprocess.run(
        ["fc-list", pattern, "file", "index", "postscriptname", "family", "familylang"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    fonts = []
    for line in res.splitlines():
        if line.startswith(":"):
            continue
        path, rest = line.split(": ", 1)
        family, rest = rest.split(":index=")
        if ":" in family:
            family, familylang = family.split(":familylang=")
        else:
            familylang = ""
        if family.startswith("."):
            # weird system fonts => skip
            continue
        if ":postscriptname=" not in rest:
            # weird unnamed variable font instance => skip
            continue
        index, name = rest.split(":postscriptname=")
        index = int(index)
        if index >= 65536:
            # weird variable font indexing => skip
            continue
        name = name.replace("\\", "")
        family = family.replace("\\", "").split(",")[0]
        familylangs = familylang.split(",")
        fonts.append(Font(name, path, index, family, familylangs))
    return deduplicate_fonts(fonts)


def deduplicate_fonts(fonts: list[Font]) -> list[Font]:
    """Deduplicate fonts by font name."""
    return sorted(set(fonts))


def list_text_fonts(pattern: str = ":") -> list[Font]:
    """List all fonts meant for typesetting text.
    :returns: list of fonts
    """
    fonts = list_fonts(pattern)
    # remove non-text Google fonts
    fonts = filter_google_fonts_by_category(fonts, "/Not text/Symbols", negate=True)
    # remove symbol Google fonts
    fonts = filter_google_fonts_by_metadata_match(
        fonts, 'classifications: "SYMBOLS"', negate=True
    )
    return fonts


@lru_cache(maxsize=None)
def get_charset(path, index=0) -> str:
    """Read the supported charset from the font file."""
    chars = []
    res = subprocess.run(
        ["fc-query", "--format=%{charset}\n", path, f"--index={index}"],
        check=True,
        capture_output=True,
        text=True,
    )
    for value in res.stdout.split():
        if "-" in value:
            start, end = value.split("-")
            for cp in range(int(start, 16), int(end, 16) + 1):
                chars.append(chr(cp))
        else:
            chars.append(chr(int(value, 16)))
    return "".join(sorted(chars))


def list_latin_fonts() -> list[Font]:
    """List Latin fonts for use with fontspec."""
    # fonts with latin primary_script
    google_fonts1 = filter_google_fonts_by_metadata_match(
        list_text_fonts(), f'primary_script: "Latn"'
    )
    # fonts without primary_script but latin subset
    google_fonts2 = filter_google_fonts_by_subset(
        filter_google_fonts_by_metadata_match(
            list_text_fonts(), "primary_script:", negate=True
        ),
        "latin",
    )
    google_fonts = deduplicate_fonts(google_fonts1 + google_fonts2)
    other_fonts = list_text_fonts(":lang=de|en|es|fi|fr|is|it|nl|no|pl|pt|sv")
    fonts = deduplicate_fonts(google_fonts + other_fonts)
    assert fonts, "Error finding Latin fonts"
    return fonts


def list_simplified_chinese_fonts() -> list[Font]:
    """List Chinese fonts for use with fontspec."""
    # https://fonts.google.com/?lang=zh_Hans
    google_fonts = filter_google_fonts_by_primary_script(list_text_fonts(), "Hans")
    other_fonts = list_text_fonts(":lang=zh-cn")
    fonts = deduplicate_fonts(google_fonts + other_fonts)
    # exclude other than simplified Chinese fonts
    fonts = [
        font
        for font in fonts
        if (
            "zh-cn" in font.familylangs
            or (not any(lang.startswith("zh-") for lang in font.familylangs))
        )
        and " HK" not in font.family
        and " JP" not in font.family
        and " KR" not in font.family
        and " TC" not in font.family
        and " TW" not in font.family
    ]
    assert fonts, "Error finding Chinese fonts"
    return fonts


def list_japanese_fonts() -> list[Font]:
    """List Japanese fonts for use with fontspec."""
    # https://fonts.google.com/?lang=ja_Jpan
    google_fonts = filter_google_fonts_by_primary_script(list_text_fonts(), "Jpan")
    other_fonts = list_text_fonts(":lang=ja")
    fonts = deduplicate_fonts(google_fonts + other_fonts)
    assert fonts, "Error finding Japanese fonts"
    return fonts


def list_korean_fonts() -> list[Font]:
    """List Korean fonts for use with fontspec."""
    # https://fonts.google.com/?lang=ko_Kore
    google_fonts = filter_google_fonts_by_primary_script(list_text_fonts(), "Kore")
    other_fonts = list_text_fonts(":lang=ko")
    fonts = deduplicate_fonts(google_fonts + other_fonts)
    assert fonts, "Error finding Korean fonts"
    return fonts


def list_greek_fonts() -> list[Font]:
    """List Greek fonts for use with fontspec."""
    # https://fonts.google.com/?lang=el_Grek
    google_fonts = filter_google_fonts_by_subset(list_text_fonts(), "greek")
    other_fonts = list_text_fonts(":lang=el")
    fonts = deduplicate_fonts(google_fonts + other_fonts)
    assert fonts, "Error finding Greek fonts"
    return fonts


def filter_cursive_fonts(fonts: list[Font]) -> list[Font]:
    return filter_google_fonts_by_category(
        fonts,
        "/Expressive/Artistic",
        "/Expressive/Fancy",
        "/Expressive/Sophisticated",
        "/Expressive/Innovative",
        "/Script/Formal",
        "/Script/Handwritten",
        "/Script/Informal",
    )


def filter_google_fonts_by_category(fonts: list[Font], *categories: str, negate: bool = False) -> list[Font]:
    families = set()
    csv_path = os.path.join(GOOGLE_FONTS_DIR, "tags/all/families.csv")
    with open(csv_path) as f:
        for line in f:
            family, category, score = line.strip().split(",")
            if category in categories:
                families.add(family)

    filtered = [font for font in fonts if (font.family in families) != negate]
    assert filtered, "No fonts found"
    return filtered


def filter_google_fonts_by_primary_script(fonts: list[Font], script: str) -> list[Font]:
    return filter_google_fonts_by_metadata_match(fonts, f'primary_script: "{script}"')


def filter_google_fonts_by_subset(fonts: list[Font], subset: str) -> list[Font]:
    return filter_google_fonts_by_metadata_match(fonts, f'subsets: "{subset}"')


def filter_google_fonts_by_metadata_match(
    fonts: list[Font], pattern: str, *, negate: bool = False
) -> list[Font]:
    dirs = set()
    metadata_glob = os.path.join(GOOGLE_FONTS_DIR, "**/METADATA.pb")
    for path in glob.glob(metadata_glob, recursive=True):
        with open(path) as f:
            txt = f.read()
        if pattern in txt:
            dirs.add(os.path.dirname(path))
    fonts = [
        font
        for font in fonts
        if any(font.path.startswith(dirname + "/") for dirname in dirs) != negate
    ]
    return fonts


# import re
# import subprocess
#
# for name, package in sorted(PDFLATEX_FONTS.items()):
#     package = re.sub(r"\s*%.*\n", r"\n", package).strip()
#     with open("test.tex", "w") as f:
#         f.write(f"""
# \\documentclass{{article}}
# {package}
# \\begin{{document}}
# foo
# \\end{{document}}
# """)
#     res = subprocess.run(["pdflatex", "-interaction=batchmode", "test.tex"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#     print("    ", end="")
#     if res.returncode != 0:
#         print("# ", end="")
#     print(f"\"{name}\": {repr(package)},")
