# Data Dictionary

| Variable | Definition | Unit | Levels | Type
|----------|------------|------|--------|------
`PassengerId` | Unique identifier for passengers |  |  | Positive integer
`Survived` | Whether the passenger survived |  | `0` = No<br>`1` = Yes | Boolean
`Pclass` | Ticket class |  | `1` = 1st (Upper class)<br>`2` = 2nd (Middle class)<br>`3` = 3rd (Lower class) | Categorical
`Sex` | Sex |  | `male`<br>`female` | Categorical
`Age` | Age | years |  | Numeric
`SibSp` | # of siblings + spouses aboard |  |  | Non-negative integer
`Parch` | # of parents + children aboard |  |  | Non-negative integer
`Ticket` | Ticket number |  |  | String
`Fare` | Fare paid | dollars(?) |  | Numeric
`Cabin` | Cabin number |  |  | String
`Embarked` | Port of embarkation |  | `C` = Cherbourg<br>`Q` = Queenstown<br>`S` = Southampton | Categorical

## Notes

#### `age`

Age is fractional if less than 1. If the age was estimated, it is in the form of `xx.5`.

#### `sibsp` and `parch`

The dataset defines family relations in the following way:

<dl>
    <dt>Sibling</dt>
    <dd>A brother, sister, stepbrother, or stepsister.</dd>
    <dt>Spouse</dt>
    <dd>A husband or wife. Fiancés and mistresses were ignored.</dd>
    <dt>Parent</dt>
    <dd>A mother or father.</dd>
    <dt>Child</dt>
    <dd>A son, daughter, stepson, or stepdaughter.</dd>
</dl>

Some children traveled with only a nanny. These children have `parch = 0`.