# Long Media Cameras — краткое руководство

**Long Media Cameras** управляет направлением камеры независимо от scene content.

## Рекомендуемое подключение

```text
Long Media Planner
        ↓ clip_plan
Long Media Cameras
        ↓ clip_plan
Long Media Setup
```

В **Long Media Setup** выберите:

```text
Timeline Mode: multiclip
```

Cameras можно использовать standalone, но Planner → Cameras → Setup — рекомендуемый MultiClip workflow.

## Auto Sync Planner

Оставляйте **Auto Sync Planner = ON**, если подключён Planner.

Тогда Cameras:

- создаёт одну camera card на каждый Planner clip;
- связывает cards через стабильный `clip_id`;
- следует за reorder клипов Planner;
- сохраняет camera settings при перемещении клипов;
- автоматически отключает `Transition to Next` на последнем clip.

## Параметры camera card

Каждая card управляет одним clip:

- **Shot Size** — framing и camera distance;
- **Rig / Support** — физический support/capture character;
- **Camera Body** — характер camera/sensor;
- **Lens** — family/focal length;
- **Stabilization** — характер стабилизации;
- **Movement Path** — camera trajectory;
- **Movement Intensity** — скорость/сила движения;
- **Transition Type** — отношение к следующему clip;
- **Space Relation** — тот же, соседний или другой space;
- **Entity Continuity** — насколько сильно люди и layout должны оставаться пространственно согласованными;
- **Transition to Next** — включает transition contract к следующей card.

## Transition Type

### Continuous / Same Shot
Используйте, когда следующий clip должен ощущаться продолжением одного непрерывного camera shot. Это лучший default для длинных continuous sequences.

### Threshold Entry
Для движения через физическую границу: doorway, arch, gate, corridor entrance, tunnel и т. п.

### Occluded Hidden Cut
Использует visual occlusion как boundary. Подходит, когда нужно сохранить ощущение continuity, но разрешить скрытый editorial transition.

### Hard Cut
Намеренный видимый монтажный cut. Обычно сочетается с **Different Space** при смене локации.

## Camera presets

Доступны sequence presets:

- Continuous Push-In
- Reveal Pull-Back
- Ritual Orbit
- Lateral Reveal
- Descent Into Scene
- Slow Cinematic Drift
- Static Tension → Push
- Approach → Threshold → Interior

Preset заполняет camera cards всей sequence; после этого каждую card можно редактировать вручную.

## Владение prompt

Когда подключён **Long Media Cameras**, позвольте Cameras владеть operator language.

Planner prompts должны описывать в основном:

- персонажей;
- действия и body movement;
- environment;
- lighting;
- atmosphere;
- object interaction;
- развитие сцены.

Cameras компилирует собственные camera instructions в Planner output и удаляет конфликтующие camera directives из Planner text. Разделение scene content и camera direction даёт наиболее чистый control.

## Пример

Planner:

```text
clip_1:
The woman walks slowly through the empty ceremonial hall.

clip_2:
She continues forward and gradually raises her right hand toward the altar.

clip_3:
Her hand reaches the glowing surface and the surrounding symbols activate.
```

Cameras:

```text
Clip 1: Wide Shot · Track Forward · Slow
Clip 2: Medium Shot · Track Forward · Slow
Clip 3: Medium Close-Up · Push-In · Ultra Slow
Transition: Continuous / Same Shot
Space: Same Space
Entity Continuity: Lock Population / Layout
```
