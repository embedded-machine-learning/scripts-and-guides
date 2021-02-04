#/bin/bash

echo "sdfsdf"

cat > README.md << "EOL"

# Embedded Machine Learning Scripts and Guides

This is a collection of scripts and guides by the <a href="https://embedded-machine-learning.github.io/webpage/">Christian Doppler Laboratory for Embedded Machine Learning</a>

We collect scripts and guides that help us in our everyday work to setup software and frameworks. This repository is also the source of an EML Toolbox that aims to easyily implement machine learning toolchains.

## Table of Contents

### Guides

EOL

#  loop through dir
for d in guides/*.md; do
    dirname=$(basename ${d%/})
    #  remove numbers
    name_print=${dirname}
    # printf "\n\n## ${name_print//_/ }\n" >> README.md
    grep -m 1 "#" guides/${dirname}

    for d2 in ${d}/*.md; do
        name=$(basename ${d2%.md})
        printf "* [" >> README.md
        grep -m 1 "#" guides/${dirname} | tr '\n' '\0' | tr '#' '\0'>> README.md
        printf "](./guides/${dirname})\n" >> README.md

    done
done

printf "\n\n### Scripts\n" >> README.md


#  loop through dir
for d in scripts/*/; do
    dirname=$(basename ${d%/})
    #  remove numbers
    name_print=${dirname}
    #printf "\n\n## ${name_print//_/ }\n" >> README.md

    for d2 in ${d}/*.md; do
        name=$(basename ${d2%.md})
        printf "* [" >> README.md
        grep -m 1 "#" scripts/${dirname}/README.md | tr '\n' '\0' | tr '#' '\0'>> README.md
        printf "](./scripts/${dirname}/README.md)\n" >> README.md
        #printf "* [${name//_/ }](./scripts/${dirname}/${name}.md/)\n" >> README.md

    done
done

    #printf "\n\nTo automatically rebuild the README file on each commit, run "\""bin/activate_hook"\"" from inside the repo once." >> ${dir}/README.md

git add README.md


echo "README updated!"

cat README.md

git status
