package com.smartsupply.backend;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class CalendarDTO {
    private String calendarDate;
    private int dayOfWeek;
    private int isHoliday;
    private String eventType;
    private int isPayloadDay;
}