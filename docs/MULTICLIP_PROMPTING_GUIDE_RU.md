# Руководство по prompting для MultiClip

Главное правило:

**Global Prompt описывает то, что остаётся постоянным.**

**Clip Prompt описывает то, что меняется во времени.**

## Global Prompt

Помещайте постоянную информацию о сцене в Global Prompt.

```text
A pale woman in a dark ceremonial robe stands inside an enormous ancient
sci-fi temple. Cold metallic architecture, monumental scale, dim amber ritual
light, realistic materials, cinematic dark atmosphere.
```

Это общий world state для всей sequence.

## Clip prompts

Каждый следующий clip должен продолжать состояние, созданное предыдущим.

```text
clip_1:
The woman slowly walks toward the central altar. Her robe moves naturally with
each step. The surrounding crowd remains still and attentive.

clip_2:
She continues the same walk and gradually raises her right hand. The ritual
lights begin pulsing softly across the walls.

clip_3:
Her raised hand reaches the altar surface. The symbols surrounding it gradually
activate and fill the chamber with warm light.
```

## Полезные слова continuity

Подходящие temporal phrases:

- `continues`
- `keeps moving`
- `gradually`
- `slowly`
- `the movement develops`
- `the same action continues`
- `reaches`
- `begins`
- `moves closer`
- `transitions into`

Они дают следующему clip понятное motion/state, которое нужно унаследовать.

## Перенос действия через границу клипа

Для плавного handoff позволяйте действию естественно пересекать boundary.

```text
clip_2:
She gradually raises her right hand toward the glowing surface.

clip_3:
Her right hand continues the same movement and gently touches the glowing surface.
```

Второй prompt начинается из уже установленного направления движения.

## Prompting с Long Media Cameras

Когда подключён **Long Media Cameras**, camera direction оставляйте в Cameras.

Clip prompts должны описывать:

- character actions;
- body movement;
- environment;
- lighting changes;
- atmosphere;
- object interaction;
- scene progression.

Пример:

```text
clip_1:
The man walks slowly through the crowded street.

clip_2:
He continues forward and turns his head toward the neon signs.

clip_3:
He reaches the entrance, stops beside it and looks inside.
```

Shot size, lens, movement path, stabilization и transition behavior задаёт Cameras.

## Prompting без Cameras

Если Cameras не подключён, operator direction можно писать непосредственно в prompt каждого clip.

```text
clip_1:
The woman walks through the hall. A wide frontal tracking shot moves slowly backward.

clip_2:
She continues walking as the camera smoothly arcs toward her left side.

clip_3:
The continuous camera movement gradually approaches a medium close framing.
```

## Стиль формулировок MiniMax-H3

Предпочитайте позитивные описания желаемого visual state.

```text
The architecture remains stable and preserves its original proportions.
The same characters remain in their established positions.
The movement continues smoothly at the same slow pace.
```

Позитивные state descriptions особенно полезны для continuity, identity, architecture и медленного camera motion.

## Audio и lip-sync wording

При `audio_mode=lip_sync` подключите авторитетную performance к `audio_1` и семантически опишите видимую performance.

```text
She continues singing with clear natural mouth articulation synchronized to
Audio 1. Her body movement remains slow and cinematic.
```

LongMedia получает timing из подключённого audio source; prompt должен описывать желаемое визуальное исполнение, а не вручную раскладывать фонемы по времени.

## Structured MultiClip prompt

Для Planner import:

```text
clip_1:
...

clip_2:
...

clip_3:
...
```

Число `clip_N` секций должно соответствовать структуре ролика. Duration и Seed остаются редактируемыми внутри Planner cards.
